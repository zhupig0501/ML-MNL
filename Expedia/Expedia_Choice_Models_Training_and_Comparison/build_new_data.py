import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter

def generate_data(order_num,product_num):
    df = pd.read_csv("df_OR_normal_process_with_nonpurchase_clean.csv")
    df = df.drop(columns = ['prop_id', 'srch_length_of_stay','srch_id','booking_bool'])
    features = [i for i in df.columns]


    features_pool = {}
    for i in features:
        features_pool[i] = df[i].unique()
    

    next_train_data = []
    for i in range(order_num):
        for j in range(product_num):
            new_row = [i]
            for feature in features:
                new_row.append(np.random.choice(features_pool[feature]))
            next_train_data.append(new_row)
    
    next_train_data = pd.DataFrame(next_train_data,columns = ['srch_id']+features)
    next_train_data = next_train_data.astype('float')

    return next_train_data

