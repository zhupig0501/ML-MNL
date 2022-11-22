import numpy as np
import pandas as pd 
from torch import nn
from choicemodels import MNL_loss,MLE,DeepCell,DeepCell_With_Assortment,EarlyStopping
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ChoiceModel import ChoiceModel


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


