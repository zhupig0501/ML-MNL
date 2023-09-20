import numpy as np
import pandas as pd 
from torch import nn
from choicemodels_real_data import MNL_loss,MLE,DeepCell,EarlyStopping,DeepCell_With_Assortment,Exp_loss,MMNL_loss
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from scipy.optimize import minimize,Bounds
#from pytorchtools import Earlystopping


def train_data(training_data, label, train_dic, BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.01,path = 'Assortment\\DeepFM_parameters_ztn.pt',patience = 10,early_stop_num = 5,model_type = 'MNL'):
    total = len(train_dic)
    total_assort_id_list = list(train_dic)
    early_stopping = EarlyStopping(save_path = path, patience=patience)
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLE().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    if model_type == 'MNL':
        criterion = MNL_loss().to(device)
    else:
        criterion = Exp_loss().to(device)
    for epoch in range(NUM_EPOCHS):
        total_assortment_list = []
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        assortments = np.random.permutation(total)
        # tmp_data = [training_data[0][indices], training_data[1][indices], label[indices]]
        batches = list()
        # i = 0
        # while i < total:
        #     j = min(i + BATCH_SIZE, total)
        #     batches.append([tmp_data[0][i: j], tmp_data[1][i: j], tmp_data[2][i: j]])
        #     i = j
        i = 0
        while i < total:
            j = min(i + BATCH_SIZE, total) - 1
            temp_list = []
            temp_assortment_list = []
            count = 0
            for item in range(i, j+1):
                temp_list.extend(train_dic[total_assort_id_list[assortments[item]]])
                temp_len = len(train_dic[total_assort_id_list[assortments[item]]])
                temp_lst = []
                for i in range(count, temp_len + count):
                    temp_lst.append(i)
                temp_assortment_list.append(temp_lst)
                count += temp_len
            batches.append([training_data[0][temp_list], training_data[1][temp_list], label[temp_list]])
            total_assortment_list.append(temp_assortment_list)
            i = j + 1
        for step, batch in enumerate(batches):
            model.train() 
            optimizer.zero_grad()
            #pdb.set_trace()
            outputs = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
            # print(outputs)
            y = torch.from_numpy(batch[2]).to(torch.float).to(device)
            #print(y)
            loss = criterion(outputs, y, total_assortment_list[step])
            #print(loss)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            loss = 0
            for i in range(20):
                assortments = np.random.permutation(total)
                temp_list = []
                temp_assortment_list = []
                count = 0
                for item in range(0, 50):
                    temp_list.extend(train_dic[total_assort_id_list[assortments[item]]])
                    temp_len = len(train_dic[total_assort_id_list[assortments[item]]])
                    temp_lst = []
                    for i in range(count, temp_len + count):
                        temp_lst.append(i)
                    temp_assortment_list.append(temp_lst)
                    count += temp_len
                cat = torch.from_numpy(training_data[0][temp_list]).to(torch.long).to(device)
                weight = torch.from_numpy(training_data[1][temp_list]).to(torch.float).to(device)
                outputs = model([cat, weight])
                y = torch.from_numpy(label[temp_list]).to(torch.float).to(device)
                loss += criterion(outputs,y, temp_assortment_list).item()
            print("loss {:g}".format(loss / 20))
            if epoch+1 >= early_stop_num:
                early_stopping(loss, model)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
    return model

def train_DeepFM(training_data,label,valid_data,valid_label,valid_dic,BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.01,weight1 = 30,patience = 10,path = 'Assortment\\DeepFM_parameters_ztn.pt',early_stop_num = 5):
    # device = "cpu"
    total = training_data.shape[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepCell().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(save_path = path, patience=patience)
    for epoch in range(NUM_EPOCHS):
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        indices = np.random.permutation(total)
        tmp_data = [training_data[0][indices], training_data[1][indices], label[indices]]
        batches = list()
        i = 0
        while i < total:
            j = min(i + BATCH_SIZE, total)
            batches.append([tmp_data[0][i: j], tmp_data[1][i: j], tmp_data[2][i: j]])
            i = j
        for step, batch in enumerate(batches):
            model.train() 
            optimizer.zero_grad()
            #pdb.set_trace()
            outputs = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
            # print(outputs)
            y = torch.from_numpy(batch[2]).to(torch.float).to(device)
            # print(y)
            class_weight = Variable(torch.FloatTensor([1, weight1])).to(device)
            weight_variable = class_weight[y.long()] 
            criterion = nn.BCEWithLogitsLoss(weight = weight_variable).to(device)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            cat = torch.from_numpy(valid_data[0]).to(torch.long).to(device)
            weight = torch.from_numpy(valid_data[1]).to(torch.float).to(device)
            outputs = model([cat, weight])
            y = torch.from_numpy(valid_label).to(torch.float).to(device)
            class_weight = Variable(torch.FloatTensor([1, weight1])).to(device)
            weight_variable = class_weight[y.long()] 
            criterion = nn.BCEWithLogitsLoss(weight = weight_variable).to(device)
            loss = criterion(outputs, y).item()
            
            outputs = np.array(outputs.cpu())
            pro = np.exp(outputs)/(np.exp(outputs)+1)
            total_acc = 0
            count = 0
            for item in valid_dic:
                temp_prob_max = []
                temp_prob = list(pro[valid_dic[item]])
                temp_test_y = list(valid_label[valid_dic[item]])
                total_acc += 1
                temp_prob_max.append(temp_prob.index(max(temp_prob)))
                temp_y_max = temp_test_y.index(max(temp_test_y))
                if temp_y_max in temp_prob_max:
                    count += 1
                
            print("loss {:g},acc {:g}".format(loss,count/total_acc))
            if epoch+1 >= early_stop_num:
                early_stopping(loss, model)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
    return model

def train_DeepFMa(training_data,label,valid_data,valid_label,valid_dic,BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.01,weight1 = 30,patience = 10,path = 'Assortment\\DeepFM_parameters_ztn.pt',early_stop_num = 5):
    # device = "cpu"
    total = training_data.shape[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepCell_With_Assortment().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(save_path = path, patience=patience)
    for epoch in range(NUM_EPOCHS):
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        indices = np.random.permutation(total)
        tmp_data = [training_data[0][indices], training_data[1][indices], label[indices]]
        batches = list()
        i = 0
        while i < total:
            j = min(i + BATCH_SIZE, total)
            batches.append([tmp_data[0][i: j], tmp_data[1][i: j], tmp_data[2][i: j]])
            i = j
        for step, batch in enumerate(batches):
            model.train() 
            optimizer.zero_grad()
            #pdb.set_trace()
            outputs = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
            # print(outputs)
            y = torch.from_numpy(batch[2]).to(torch.float).to(device)
            # print(y)
            class_weight = Variable(torch.FloatTensor([1, weight1])).to(device)
            weight_variable = class_weight[y.long()] 
            criterion = nn.BCEWithLogitsLoss(weight = weight_variable).to(device)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            cat = torch.from_numpy(valid_data[0]).to(torch.long).to(device)
            weight = torch.from_numpy(valid_data[1]).to(torch.float).to(device)
            outputs = model([cat, weight])
            y = torch.from_numpy(valid_label).to(torch.float).to(device)
            class_weight = Variable(torch.FloatTensor([1, weight1])).to(device)
            weight_variable = class_weight[y.long()] 
            criterion = nn.BCEWithLogitsLoss(weight = weight_variable).to(device)
            loss = criterion(outputs, y).item()
            
            outputs = np.array(outputs.cpu())
            pro = np.exp(outputs)/(np.exp(outputs)+1)
            total_acc = 0
            count = 0
            for item in valid_dic:
                temp_prob_max = []
                temp_prob = list(pro[valid_dic[item]])
                temp_test_y = list(valid_label[valid_dic[item]])
                total_acc += 1
                temp_prob_max.append(temp_prob.index(max(temp_prob)))
                temp_y_max = temp_test_y.index(max(temp_test_y))
                if temp_y_max in temp_prob_max:
                    count += 1
                
            print("loss {:g},acc {:g}".format(loss,count/total_acc))
            if epoch+1 >= early_stop_num:
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  
    return model

#core function for minimize
def func(x,model_list,data,label,d,y,train_dic):
    total = len(train_dic)
    dic_list = list(train_dic)
    z = y + x[0]*d
    results = 0
    pro = np.zeros(total)
    for i,m in enumerate(model_list):
        with torch.no_grad():
            outputs= m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
            outputs = np.exp(outputs.numpy())
            for j,item in enumerate(dic_list):
                pro[j] = pro[j] + z[i]*np.sum(outputs[train_dic[item]]*label[train_dic[item]])/(np.sum(outputs[train_dic[item]]))
    results = results + np.sum(np.log(pro))
        
    return -results/total

#A modified Frank-Wolfe algorithm
def MFW(model_list,data,label,train_dic):
    total = len(train_dic)
    dic_list = list(train_dic)
    variable_num = len(model_list)
    
    #inital solution
    x = np.zeros(variable_num)
    for i in range(variable_num):
        x[i] = 1/variable_num

    k = 1
    while k < 100:
        #calculate the derivative of function when y is given and calculate the best z 
        grad_value = np.zeros(variable_num)
        pro_list = []
        pro = np.zeros(total)
        for i,m in enumerate(model_list):
            pro_temp = np.zeros(total)
            with torch.no_grad():
                outputs = m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
                outputs = np.exp(outputs.numpy())
                for j,item in enumerate(dic_list):
                    pro[j] = pro[j] + x[i]*np.sum(outputs[train_dic[item]]*label[train_dic[item]])/(np.sum(outputs[train_dic[item]]))
                    pro_temp[j] = pro_temp[j] + x[i]*np.sum(outputs[train_dic[item]]*label[train_dic[item]])/(np.sum(outputs[train_dic[item]]))
                pro_list.append(pro_temp)
        for idx,value in enumerate(pro_list):
            grad_value[idx] = np.sum(value/pro) + grad_value[idx]
        grad_value = -grad_value/total
        
        #Step 1: Toward Step
        z_t = np.zeros(variable_num)
        idx = np.argmin(grad_value)
        z_t[idx] = 1
        d_t = z_t - x
        
        #Step 2: Away Step
        z_a = np.zeros(variable_num)
        idx_a = np.argsort(grad_value)[::-1]
        for idx in idx_a:
            if x[idx] != 0:
                z_a[idx] = 1
                break
        d_a = x - z_a
        
        ub_a = 0
        for i in range(x.shape[0]):
            if d_a[i] != 0 and -x[i]/d_a[i] > ub_a:
                ub_a = -x[i]/d_a[i]
       
        # Step 3: Choosing a descent direction 
        if np.dot(d_t,grad_value) <= np.dot(d_a,grad_value):
            d = d_t
            ub = 1
        else:
            d = d_a
            ub = ub_a

        #Step 4: Line search
        bound = Bounds(0,ub)
        res = minimize(func,np.array(0),args = (model_list,data,label,d,x,train_dic),bounds=bound,method='trust-constr')
        gamma = res.x[0]
        
        print(gamma)
        #Step 5: Update
        if gamma < 0.005:
            break
        else:
            x = x + gamma*d
        k = k + 1
    return x

# calculate the in-sample loss given a model list
def cal_loss(model_list,data,label,alpha,train_dic,total):
    dic_list = list(train_dic)
    loss = 0
    pro = np.zeros(total)
    for i,m in enumerate(model_list):
        with torch.no_grad():
            outputs = m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
            outputs = np.exp(outputs.numpy())
            for j,item in enumerate(dic_list):
                pro[j] = pro[j] + alpha[i]*np.sum(outputs[train_dic[item]]*label[train_dic[item]])/(np.sum(outputs[train_dic[item]]))
    loss = loss + np.sum(np.log(pro))
    
    return -loss/total


# Estimate the LC-MNL
def CG_algo(training_data,label,train_dic,BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.005,patience = 5,early_stop_num = 5,path = "trained_model_ztn\\LC-MNL_parameters_ztn.pt"):
    total = len(train_dic)
    total_assort_id_list = list(train_dic)
    iterate_times = 100
    
    #set MNL model as initial solution
    model_list = [torch.load("trained_model_ztn\\MNL_parameters_ztn.pt")]
    alpha = np.array([1])

    #calculate the current MNL loss
    loss_1 = cal_loss(model_list,training_data,label,alpha,train_dic,total)
    
    for k in range(iterate_times):

        #Solving the Support-Finding Step using SGD
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MLE().to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR)
        criterion = MMNL_loss().to(device)
        early_stopping = EarlyStopping(save_path = path, patience=patience)
        for epoch in range(NUM_EPOCHS):
            total_assortment_list = []
            print(f'---------------- Epoch: {epoch+1:02} ----------')
            assortments = np.random.permutation(total)
            # tmp_data = [training_data[0][indices], training_data[1][indices], label[indices]]
            batches = list()
            # i = 0
            # while i < total:
            #     j = min(i + BATCH_SIZE, total)
            #     batches.append([tmp_data[0][i: j], tmp_data[1][i: j], tmp_data[2][i: j]])
            #     i = j
            i = 0
            while i < total:
                j = min(i + BATCH_SIZE, total) - 1
                temp_list = []
                temp_assortment_list = []
                count = 0
                for item in range(i, j+1):
                    temp_list.extend(train_dic[total_assort_id_list[assortments[item]]])
                    temp_len = len(train_dic[total_assort_id_list[assortments[item]]])
                    temp_lst = []
                    for i in range(count, temp_len + count):
                        temp_lst.append(i)
                    temp_assortment_list.append(temp_lst)
                    count += temp_len
                batches.append([training_data[0][temp_list], training_data[1][temp_list], label[temp_list]])
                total_assortment_list.append(temp_assortment_list)
                i = j + 1
            for step, batch in enumerate(tqdm(batches)):
                model.train() 
                optimizer.zero_grad()
                #pdb.set_trace()
                outputs = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
                # print(outputs)
                y = torch.from_numpy(batch[2]).to(torch.float).to(device)
                # print(y)
                with torch.no_grad():
                    z = []
                    for m in model_list:
                        model_temp = m([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
                        z.append(model_temp.numpy())
                z = torch.tensor(z,requires_grad = False)
                loss = criterion(outputs, y, total_assortment_list[step],z,alpha)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                loss = 0
                for i in range(20):
                    assortments = np.random.permutation(total)
                    temp_list = []
                    temp_assortment_list = []
                    count = 0
                    for item in range(0, 50):
                        temp_list.extend(train_dic[total_assort_id_list[assortments[item]]])
                        temp_len = len(train_dic[total_assort_id_list[assortments[item]]])
                        temp_lst = []
                        for i in range(count, temp_len + count):
                            temp_lst.append(i)
                        temp_assortment_list.append(temp_lst)
                        count += temp_len
                    cat = torch.from_numpy(training_data[0][temp_list]).to(torch.long).to(device)
                    weight = torch.from_numpy(training_data[1][temp_list]).to(torch.float).to(device)
                    outputs = model([cat, weight])
                    y = torch.from_numpy(label[temp_list]).to(torch.float).to(device)
                    z = []
                    for m in model_list:
                        model_temp = m([cat,weight])
                        z.append(model_temp.numpy())
                    z = torch.tensor(z,requires_grad = False)
                    loss = criterion(outputs,y, temp_assortment_list,z,alpha).item() + loss
                print("loss {:g}".format(loss/20))
                if epoch+1 >= early_stop_num:
                    early_stopping(loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break  
        model = torch.load(path)
        model_list1 = model_list + [deepcopy(model)]

        # Solving the Proportions-Update Step
        alpha1 = MFW(model_list1,training_data,label,train_dic)
        print(alpha1)
          
        cost_diff = loss_1-cal_loss(model_list1,training_data,label,alpha1,train_dic,total)
        print(cost_diff)
        # stopping condition
        if  cost_diff > 0.001:
            alpha = alpha1
            model_list = model_list1
            loss_1 = cal_loss(model_list1,training_data,label,alpha1,train_dic,total)
        else:
            return model_list,alpha
    return model_list,alpha

'''
if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 0.01
    training_data = np.load("data1.npy")
    label = np.load("label1.npy")
    model_list,alpha = CG_algo(training_data,label,BATCH_SIZE = BATCH_SIZE,NUM_EPOCHS = NUM_EPOCHS,LR = LR)
'''