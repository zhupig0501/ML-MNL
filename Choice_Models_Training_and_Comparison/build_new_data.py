import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter


#Generate the T = order_num,, |S_t| = product_num
def generate_data(order_num,product_num):

    #Load real data
    df = pd.read_csv('data.csv')

    
    #Generarting the features pools.
    del_feature = ['orderid','alternative','orderlabel']
    features = [i for i in df.columns if i not in del_feature]
    Deptime_dict = {}
    Arrtime_dict = {}
    for i in df.index:
        Deptime_dict[df.loc[i,'outDepTime_sin']] = df.loc[i,'outDepTime_cos']
        Arrtime_dict[df.loc[i,'outArrTime_sin']] = df.loc[i,'outArrTime_cos']
    features_pool = {}
    for i in features:
        if i != 'outDepTime_sin' and i != 'outDepTime_cos' and i != 'outArrTime_sin' and i != 'outArrTime_cos':
            features_pool[i] = df[i].unique()
        elif i == 'outDepTime_sin':
            features_pool[i] = df[i].unique()
        elif i == 'outArrTime_sin':
            features_pool[i] = df[i].unique()
    
    #Generating the new datas randomly drwa from the features pools.
    next_train_data = []
    for i in range(order_num):
        od_pair = np.random.choice(features_pool['OD'])
        for k in range(product_num):
            new_row = [i]
            for feature in features:
                if feature != 'OD' and feature  != 'outDepTime_sin' and feature  != 'outDepTime_cos' and feature  != 'outArrTime_sin' and feature  != 'outArrTime_cos':
                    new_row.append(np.random.choice(features_pool[feature]))
                elif feature  == 'outDepTime_sin':
                    j = np.random.choice(features_pool[feature])
                    new_row.append(j)
                    new_row.append(Deptime_dict[j])
                elif feature  == 'outArrTime_sin':
                    j = np.random.choice(features_pool[feature])
                    new_row.append(j)
                    new_row.append(Arrtime_dict[j])
                elif feature == 'OD':
                    new_row.append(od_pair)
            next_train_data.append(deepcopy(new_row))
    next_train_data = pd.DataFrame(next_train_data,columns = ['orderid']+features)
    next_train_data = next_train_data.astype('float')

    return next_train_data

