from json.encoder import INFINITY
import numpy as np
import pandas as pd
from gurobipy import GRB,Model,quicksum
from copy import deepcopy
import torch
import multiprocessing as mt

#MNL assortment
def MNL(utility,prices,capacity = INFINITY):
    utility = np.array(utility)
    prices = np.array(prices)

    A = np.argsort(-prices)
    assortment = []
    max_r = 0
    for i in A:
        assort_utility = utility[assortment+[i]]
        assort_price = prices[assortment+[i]]
        if assort_price.dot(assort_utility)/(1+np.sum(assort_utility)) >= max_r:
            assortment.append(i)
            max_r = assort_price.dot(assort_utility)/(1+np.sum(assort_utility))
        else:
            break
    
    if capacity < len(assortment):
        if utility.size != len(set(utility)):
            utility = utility + np.random.uniform(0,0.001,size = utility.size)
        tao = []
        for i in range(utility.shape[0]):
            for j in range(i+1,utility.shape[0]+1):
                if i == 0:
                    tao.append((i-1,j-1,prices[j-1]))
                else:
                    tao.append((i-1,j-1,(utility[i-1]*prices[i-1]-utility[j-1]*prices[j-1])/(utility[i-1]-utility[j-1])))
        tao = sorted(tao,key = lambda x: x[2])
        A = np.argsort(-utility)
        B = []
        Assortment_list = [A[0:capacity]]
        for idx in tao:
            if idx[0] != -1:
                index1 = np.argwhere(A == idx[0])[0][0]
                index2 = np.argwhere(A == idx[1])[0][0]
                A[index1] = idx[1]
                A[index2] = idx[0]
            else:
                B.append(idx[1])
            G = list(A[0:capacity])
            for i in B:
                if i in G:
                    G.remove(i)
            Assortment_list.append(np.array(G))
        max_r = 0
        max_idx = 0
        for idx,num in enumerate(Assortment_list):
            assort_utility = utility[list(num)]
            assort_price = prices[list(num)]
            if assort_price.dot(assort_utility)/(1+np.sum(assort_utility)) > max_r:
                max_idx = idx
                max_r = assort_price.dot(assort_utility)/(1+np.sum(assort_utility))
        return list(Assortment_list[max_idx])
    else:
        return assortment




    return assortment



    price2 = price2/np.max(price2)
    p_max = np.max(price2)
    M,N = v2.shape
    
    Rank,MaxCap = rank(v2)
    y_lower0=np.zeros((M,N))

    y_lower1=np.zeros((M,N))
    if capacity < N:
        for i in range(M):
            for j in range(N):
                if Rank[i, j] >= capacity+ 1:
                    y_lower0[i, j] = 1 / (v0 + MaxCap[i, capacity - 1])
                else:
                    y_lower0[i, j] = 1 / (v0 + MaxCap[i, capacity] - v2[i, j])
        # 计算Y_lij1
        for i in range(M):
            for j in range(N):
                if Rank[i, j] >= capacity:
                    y_lower1[i, j] = 1 / (v0 + v2[i, j] + MaxCap[i, capacity - 2])
                else:
                    y_lower1[i, j] = 1 / (v0 + v2[i, j] + MaxCap[i, capacity- 1] - v2[i, j])

    ## 计算当capacity取到最大的时候的Y_lij0
    else:
        for i in range(M):
            for j in range(N):
                y_lower0[i, j] = 1 / (v0 + MaxCap[i, capacity - 1] - v2[i, j])
        # 计算当capacity取到最大的时候的计算Y_lij1
        for i in range(M):
            for j in range(N):
                y_lower1[i, j] = 1 / (v0 + MaxCap[i, capacity - 1])

    y_upper1 = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            y_upper1[i,j] = 1/(v0+v2[i,j])
   
    model1 = Model("conic_MC")
    model1.setParam("OutputFlag",0)
    model1.setParam(GRB.Param.TimeLimit, 600.0)
    model1.setParam(GRB.Param.Presolve, -1)
    model1.setParam(GRB.Param.Method, 1)
    model1.setParam(GRB.Param.MIPGap, 1e-200)

    x = model1.addVars(range(N),vtype = GRB.BINARY,name="x")
    y = model1.addVars(range(M),name="y")
    z = model1.addVars(range(M),range(N),name="z")
    w = model1.addVars(range(M),name="w")

    model1.update()

    model1.addConstr((quicksum(x[j] for j in range(N)) <= capacity),"capacity_con")
    model1.addConstrs((w[i] == v0+quicksum(v2[i,j]*x[j] for j in range(N)) for i in range(M)),"w_con")
    model1.addConstrs((v0*y[i]+quicksum(z[i,j]*v2[i,j] for j in range(N)) == 1 for i in range(M)),"y_con")
    model1.addConstrs((z[i, j] * w[i] >= x[j] * x[j] for i in range(M) for j in range(N)),"conic_con1")
    model1.addConstrs((y[i] * w[i] >= 1 for i in range(M)),"conic_con2")

    
    "MC inequality"
    model1.addConstrs((z[i,j]<=y_upper1[i,j]*x[j] for i in range(M) for j in range(N)),"z_con1")
    model1.addConstrs((z[i,j]>=y_lower1[i,j]*x[j] for i in range(M) for j in range(N)),"z_con2")
    model1.addConstrs((z[i,j]<=y[i]-y_lower0[i,j]*(1-x[j]) for i in range(M) for j in range(N)),"z_con3")
    model1.addConstrs((z[i,j]>=y[i]-(1/v0)*(1-x[j]) for i in range(M) for j in range(N)),"z_con4")
    
    obj = quicksum(beta[i]*v0*p_max*y[i] for i in range(M))+quicksum(beta[i]*v2[i,j]*(p_max-price2[j])*z[i,j] for i in range(M) for j in range(N))
    model1.setObjective(obj,GRB.MINIMIZE)
    model1.optimize()
    assortment = []
    for i in model1.getVars():
        if i.varName[0] == 'x' and i.x > 0.9:
            assortment.append(int(i.varName[2:-1]))
    model1.reset(0)
    return assortment
    


#Assortment for DeepFM-a
def cal_profict(model,valid_data,price,assortment,feature_column):
    valid_data = valid_data.reshape((2,1,100,17))
    temp_data = valid_data[:,:,assortment,:]
    data_temp = np.zeros((1,len(assortment),9))
    data_cate_temp = np.zeros((1,len(assortment), 2))
    min1 = np.min(temp_data[1,0,:,5])
    max1 = np.max(temp_data[1,0,:,5])
    mean1 = np.mean(temp_data[1,0,:,5])
    min2 = np.min(temp_data[1,0,:,6])
    max2 = np.max(temp_data[1,0,:,6])
    mean2 = np.mean(temp_data[1,0,:,6])
    min3 = np.min(temp_data[1,0,:,7])
    max3 = np.max(temp_data[1,0,:,7])
    mean3 = np.mean(temp_data[1,0,:,7])
    data_temp[0,:,0] = min1
    data_temp[0,:,1] = max1
    data_temp[0,:,2] = mean1
    data_temp[0,:,3] = min2
    data_temp[0,:,4] = max2
    data_temp[0,:,5] = mean2
    data_temp[0,:,6] = min3
    data_temp[0,:,7] = max3
    data_temp[0,:,8] = mean3
    for j in range(len(assortment)):
        if temp_data[1,0,j,6] == np.min(temp_data[1,0,:,6]):
            data_cate_temp[0,j,0] = 1
        if temp_data[1,0,j,6] == np.max(temp_data[1,0,:,5]):
            data_cate_temp[0,j,1] = 1
    a1 = np.concatenate([temp_data[0],np.zeros((1, len(assortment), 9)),data_cate_temp],axis = 2)
    b1 = np.concatenate([temp_data[1],data_temp,np.ones((1,len(assortment),2))],axis = 2)
    X = torch.from_numpy(a1).to(torch.long)
    weight = torch.from_numpy(b1).to(torch.float)
    utility  = model([X[:,:,feature_column],weight[:,:,feature_column]])
    pro_DeepFM = torch.sigmoid(utility).detach().numpy().reshape(len(assortment))
    profit = np.dot(pro_DeepFM,price[assortment])

    return profit

def  ml_assortment_swap(model,valid_data,price,assortment,max_assort = 100,capacity = INFINITY,feature_column = [i for i in range(28)]):
    assortment = list(assortment)
    profit = cal_profict(model,valid_data,price,assortment,feature_column)
    for k in range(10000):
        revenue = np.zeros(len(assortment))
        for idx,item in enumerate(assortment):
            temp_assort = deepcopy(assortment)
            temp_assort.remove(item)
            revenue[idx] = cal_profict(model,valid_data,price,temp_assort,feature_column)
        max_idx1 = np.argmax(revenue)
        max_idx1 = assortment[max_idx1]
        revenue = {}
        for idx in range(max_assort):
            if idx not in assortment:
                temp_assort = assortment + [idx]
                temp_assort.remove(max_idx1)
                revenue[idx] = cal_profict(model,valid_data,price,temp_assort,feature_column)
        max_idx2  = max(revenue, key = revenue.get)
        max_revunue = revenue[max_idx2]
        if profit > max_revunue:
            return assortment
        else:
            profit = max_revunue
            assortment.remove(max_idx1)
            assortment.append(max_idx2)
    return assortment








