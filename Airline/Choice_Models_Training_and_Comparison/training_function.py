import numpy as np
import pandas as pd 
from torch import nn
from choicemodels import MNL_loss,MMNL_loss,MLE,DeepCell,DeepCell_With_Assortment,EarlyStopping
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.autograd import Variable
from scipy.optimize import minimize,Bounds
from copy import deepcopy


#traing function for mnl
def train_data(training_data,label,BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.01,patience = 10,path = 'Assortment\\DeepFM_parameters_ztn.pt',early_stop_num = 5,feature_column = [i for i in range(17)]):
    total = training_data.shape[1]
    early_stopping = EarlyStopping(save_path = path, patience=patience)
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLE(feature_column = feature_column).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = MNL_loss().to(device)
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
        for step, batch in enumerate(tqdm(batches)):
            model.train() 
            optimizer.zero_grad()
            #pdb.set_trace()
            outputs,u = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
            y = torch.from_numpy(batch[2]).to(torch.float).to(device)
            loss = criterion(outputs, y,u)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            loss = 0
            for i in range(20):
                indices = np.random.permutation(total)
                cat = torch.from_numpy(training_data[0][indices][:50]).to(torch.long).to(device)
                weight = torch.from_numpy(training_data[1][indices][:50]).to(torch.float).to(device)
                outputs,u = model([cat, weight])
                y = torch.from_numpy(label[indices][:50]).to(torch.float).to(device)
                loss += criterion(outputs,y,u).item()
            print("loss {:g}".format(loss / 20))
            if epoch+1 >= early_stop_num:
                early_stopping(loss, model)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
    return model


#traing function for DeepFM and DeepFM-a models
def train_DeepFM(training_data,label,valid_data,valid_label,model_name = 'independent',BATCH_SIZE = 32,NUM_EPOCHS = 100,LR = 0.01,dropout = 0.5,h_depth = 2,deeplayer_size = 64,weight1 = 100,patience = 10,path = 'Assortment\\DeepFM_parameters_ztn.pt',early_stop_num = 5,feature_column = [i for i in range(17)]):
    total = training_data.shape[1]
    early_stopping = EarlyStopping(save_path = path, patience=patience)
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'independent':
        model = DeepCell(dropout = dropout,h_depth = h_depth,deeplayer_size = deeplayer_size,feature_column = feature_column).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR)
    else:
        model = DeepCell_With_Assortment(dropout = dropout,h_depth = h_depth,deeplayer_size = deeplayer_size,feature_column = feature_column).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR)
    
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
        for step, batch in enumerate(tqdm(batches)):
            model.train() 
            optimizer.zero_grad()
            #pdb.set_trace()
            outputs = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
            y = torch.from_numpy(batch[2]).to(torch.float).to(device)
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
            if model_name == 'attention':
                criterion = nn.CrossEntropyLoss(weight = weight_variable).to(device)
            else:
                criterion = nn.BCEWithLogitsLoss(weight = weight_variable).to(device)
            loss = criterion(outputs,y).item()
            print("loss {:g}".format(loss))
            if epoch+1 >= early_stop_num:
                early_stopping(loss, model)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
    return model


#core function for minimize
def func(x,model_list,data,label,d,y,total):
    z = y + x[0]*d
    results = 0
    pro = np.zeros(total)
    for i,m in enumerate(model_list):
        with torch.no_grad():
            outputs,u = m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
            outputs = np.exp(outputs.numpy())
            u = float(u[0])
            for j in range(total):
                if np.sum(label[j]) > 0:
                    pro[j] = pro[j] + z[i]*np.sum(outputs[j]*label[j])/(np.exp(u)+np.sum(outputs[j]))
                else:
                    pro[j] = pro[j] + z[i]*np.exp(u)/(np.exp(u)+np.sum(outputs[j]))
    results = results + np.sum(np.log(pro))
        
    return -results/total

#A modified Frank-Wolfe algorithm
def MFW(model_list,data,label):
    total = data.shape[1]
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
                outputs,u = m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
                outputs = np.exp(outputs.numpy())
                for j in range(total):
                    if np.sum(label[j]) > 0:
                        pro[j] = pro[j] + x[i]*np.sum(outputs[j]*label[j])/(np.exp(u)+np.sum(outputs[j]))
                        pro_temp[j] = pro_temp[j] + x[i]*np.sum(outputs[j]*label[j])/(np.exp(u)+np.sum(outputs[j]))
                    else:
                        pro[j] = pro[j] + x[i]*np.exp(u)/(np.exp(u)+np.sum(outputs[j]))
                        pro_temp[j] = pro_temp[j] + x[i]*np.exp(u)/(np.exp(u)+np.sum(outputs[j]))
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
        res = minimize(func,np.array(0),args = (model_list,data,label,d,x,total),bounds=bound,method='trust-constr')
        gamma = res.x[0]
        
        
        #Step 5: Update
        if gamma < 0.00001:
            print(gamma)
            break
        else:
            x = x + gamma*d
        k = k + 1
    return x

# calculate the in-sample loss given a model list
def cal_loss(model_list,data,label,alpha):
    total = data.shape[1]
    loss = 0
    pro = np.zeros(total)
    for i,m in enumerate(model_list):
        with torch.no_grad():
            outputs,u = m([torch.from_numpy(data[0]).to(torch.long), torch.from_numpy(data[1]).to(torch.float)])
            outputs = np.exp(outputs.numpy())
            u = float(u[0])
            for j in range(total):
                if np.sum(label[j]) > 0:
                    pro[j] = pro[j] + alpha[i]*np.sum(outputs[j]*label[j])/(np.exp(u)+np.sum(outputs[j]))
                else:
                    pro[j] = pro[j] + alpha[i]*np.exp(u)/(np.exp(u)+np.sum(outputs[j]))
    loss = loss + np.sum(np.log(pro))
    
    return -loss/total


# Estimate the MMNL
def CG_algo(training_data,label,BATCH_SIZE = 32,NUM_EPOCHS = 50,LR = 0.005,c_rate = 0.05):
    total = training_data.shape[1]
    iterate_times = 100
    
    #set MNL model as initial solution
    model_list = [torch.load("Assortment\\MNL_parameters_"+str(c_rate)+".pt")]
    alpha = np.array([1])

    #calculate the current MNL loss
    loss_1 = cal_loss(model_list,training_data,label,alpha)
    
    for k in range(iterate_times):

        #Solving the Support-Finding Step using SGD
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MLE().to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR)
        criterion = MMNL_loss().to(device)
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
            for step, batch in enumerate(tqdm(batches)):
                model.train() 
                optimizer.zero_grad()
                #pdb.set_trace()
                outputs,u = model([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
                y = torch.from_numpy(batch[2]).to(torch.long).to(device)
                with torch.no_grad():
                    z = []
                    u_previous = []
                    for m in model_list:
                        model_temp, u_temp = m([torch.from_numpy(batch[0]).to(torch.long).to(device), torch.from_numpy(batch[1]).to(torch.float).to(device)])
                        z.append(model_temp.numpy())
                        u_previous.append(float(u_temp[0]))
                    z = torch.tensor(z,requires_grad = False)
                    u_previous = torch.tensor(u_previous,requires_grad = False)
                loss = criterion(outputs,y,z,alpha,u_previous,u)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                cat = torch.from_numpy(training_data[0]).to(torch.long).to(device)
                weight = torch.from_numpy(training_data[1]).to(torch.float).to(device)
                outputs,u = model([cat, weight])
                z = []
                u_previous = []
                for m in model_list:
                    model_temp, u_temp = m([cat,weight])
                    z.append(model_temp.numpy())
                    u_previous.append(float(u_temp[0]))
                z = torch.tensor(z,requires_grad = False)
                u_previous = torch.tensor(u_previous,requires_grad = False)
                y = torch.from_numpy(label).to(torch.long).to(device)
                loss = criterion(outputs,y,z,alpha,u_previous,u).item()
                print("loss {:g}".format(loss))

        model_list1 = model_list + [deepcopy(model)]

        # Solving the Proportions-Update Step
        alpha1 = MFW(model_list1,training_data,label)
       
          
        cost_diff = loss_1-cal_loss(model_list1,training_data,label,alpha1)
        print(cost_diff)
    
        # stopping condition
        if  cost_diff > 0.01:
            alpha = alpha1
            model_list = model_list1
            loss_1 = loss_1 - cost_diff
        else:
            return model_list,alpha
    return model_list,alpha