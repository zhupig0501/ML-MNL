import pandas as pd
import numpy as np


def refresh_category_variable_for_embedding(df, cat_columns, mode='mean'):
    feature_sizes = []
    for cat in cat_columns:
        temp_list = list(df.groupby(cat).groups)
        # temp_dic = {}
        # for i in range(len(temp_list)):
        #     temp_dic[temp_list[i]] = i
        # df[cat] = df[cat].transform(lambda s: temp_dic[s])
        #print(cat, len(temp_list))
        if mode == '0-1':
            feature_sizes.append(len(temp_list) + 1)
        else:
            feature_sizes.append(len(temp_list))
    if mode == '0-1' or mode == 'mean':
        feature_sizes.append(2)
    return feature_sizes

def standardize_continuous_variable(df, conti_columns):
    pass
    # for conti in conti_columns:
    #     if max(df[conti]) - min(df[conti]) != 0:
    #         df[conti] = df[conti].transform(lambda s:(s-min(s))/(max(s)-min(s)))

def generate_new_outside_option_with_mean(df, cont_cols_idx, cat_cols_idx, id_column, label, id_value):
    # Calculate mean of continuous variables and mode of category variable
    mean_cont_cols = df.iloc[:, cont_cols_idx].mean()
    mode_cat_cols = df.iloc[:, cat_cols_idx].mode().iloc[0]
    # Set outside_option to 0 for existing rows
    df['outside_option'] = 0
    # Create new row with outside option
    new_row = mean_cont_cols.append(mode_cat_cols).to_dict()
    new_row[id_column] = id_value
    # Check if all labels in the assortment are 0
    if (df[label] == 0).all():
        new_row[label] = 1
    else:
        new_row[label] = 0
    new_row['outside_option'] = 1
    df = df.append(new_row, ignore_index=True)
    return df

def generate_new_outside_option_with_0_and_minus1(df, conti_columns, cat_columns, id_column, label, id_value, feature_size_list):
    # Set outside_option to 0 for existing rows
    df['outside_option'] = 0
    # Create new row with outside option
    new_row = pd.Series(0, index=conti_columns)
    for cat_column in cat_columns:
        temp_index = cat_columns.index(cat_column)
        new_row = new_row.append(pd.Series(feature_size_list[temp_index]-1, index=[cat_column]))
    new_row[id_column] = id_value
    new_row['outside_option'] = 1
    # Check if all labels in the assortment are 0
    if (df[label] == 0).all():
        new_row[label] = 1
    else:
        new_row[label] = 0
    df = df.append(new_row, ignore_index=True)
    
    return df


def generate_three_lists_for_numpy_array(df, total_list, orderid, orderlabel, mode, conti_columns, cat_columns, feature_sizes):
    df_list=[]
    y_list=[]
    valid_len_list = []
    conti_index = generate_continuous_index(df, conti_columns)
    cat_index = generate_category_index(df, cat_columns)
    for item in total_list:
        temp_df=df[df[orderid]==item]
        temp_df=temp_df.reset_index(drop=True)
        if mode == 'mean':
            temp_df = generate_new_outside_option_with_mean(temp_df, conti_index, cat_index, orderid, orderlabel, item)
        if mode == '0-1':
            temp_df = generate_new_outside_option_with_0_and_minus1(temp_df, conti_columns, cat_columns, orderid, orderlabel, item, feature_sizes)
        valid_len = len(temp_df)
        valid_len_list.append(valid_len)
        y_list.append(np.array(temp_df[orderlabel]))
        temp_df.drop(columns=[orderid, orderlabel],inplace=True)
        df_list.append(temp_df)
    return df_list, y_list, valid_len_list

def generate_two_lists_for_numpy_array(df, total_list, orderid, mode, conti_columns, cat_columns, feature_sizes):
    df_list=[]
    valid_len_list = []
    conti_index = generate_continuous_index(df, conti_columns)
    cat_index = generate_category_index(df, cat_columns)
    for item in total_list:
        temp_df=df[df[orderid]==item]
        temp_df=temp_df.reset_index(drop=True)
        valid_len = len(temp_df)
        valid_len_list.append(valid_len)
        temp_df.drop(columns=[orderid, orderlabel],inplace=True)
        df_list.append(temp_df)
    return df_list, valid_len_list

def generate_continuous_index(df, conti_columns):
    conti_index=[]
    for column in df.columns:
        if column in conti_columns:
            conti_index.append(df.columns.get_loc(column))
    return conti_index

def generate_category_index(df, cat_columns):
    cat_index=[]
    for column in df.columns:
        if column in cat_columns:
            cat_index.append(df.columns.get_loc(column))
    return cat_index

def generate_numpy_array(df_list, y_list, valid_len_list, total_list, conti_index, cat_index, max_assortment_size, mode):
    total_sample_length = len(df_list)
    for i in range(total_sample_length):
        df_list[i] = np.array(df_list[i])
    #print(total_sample_length, max_assortment_size, len(cat_index))
    np_cat = np.zeros([total_sample_length, max_assortment_size, len(cat_index)])
    np_conti = np.zeros([total_sample_length, max_assortment_size, len(conti_index)])
    if mode == "+u":
        np_y = np.zeros([total_sample_length, max_assortment_size+1])
    else:
        np_y = np.zeros([total_sample_length, max_assortment_size])
    np_purchase = np.zeros([total_sample_length, 1])
    for i in range(len(total_list)):
        valid_len = valid_len_list[i]
        np_cat[i, 0:valid_len, :] = df_list[i][:, cat_index]
        np_conti[i, 0:valid_len, :] = df_list[i][:, conti_index]
        np_y[i, 0:valid_len] = y_list[i][:]
        np_purchase[i, 0] = int(np.sum(y_list[i]))

        if mode == "+u":
            if sum(y_list[i])==0:
                np_y[i,-1] = 1
    np_valid_len = np.array(valid_len_list)
    #print("purchase_rate:",sum(np_purchase)/len(total_list))
    return np_conti, np_cat, np_y, np_purchase, np_valid_len

def transfer_dataframe_into_npy(df, orderid, orderlabel, conti_columns, cat_columns, data_name='data', choice_label_name='choice_label', purchase_label_name='purchase_label', valid_length_name='valid_len', feature_size_list_name='category_feature_sizes', max_num=None, mode='mean',train_flag = True):
    #df = pd.read_csv(csvname)
    if not train_flag:
        df[orderlabel] = 0
    if mode == "choice" and train_flag:
        grouped = df.groupby(orderid)
        assortments_to_drop = []
        for group_name, group_df in grouped:
            if sum(group_df[orderlabel]) == 0:
                assortments_to_drop.append(group_name)
        df = df[~df[orderid].isin(assortments_to_drop)]
        df = df.reset_index(drop=True)
    temp_df=df.groupby(orderid).agg({orderid:np.size})
    total_list = list(temp_df.index)
    # refresh_category_variable_for_embedding
    feature_sizes = refresh_category_variable_for_embedding(df, cat_columns, mode)
    # print(feature_sizes)
    # print('finish refresh_category_variable_for_embedding')
    if max_num == None:
        max_num = max(temp_df[orderid])
        if mode == 'mean' or mode == '0-1':
            max_num += 1
    ### temp
    max_num = int(max_num)
    # standardize_continuous_variable
    standardize_continuous_variable(df, conti_columns)
    #print('finish standardize_continuous_variable')
    df_list, y_list, valid_len_list = generate_three_lists_for_numpy_array(df, total_list, orderid, orderlabel, mode, conti_columns, cat_columns, feature_sizes)
    #print('finish generate_three_lists_for_numpy_array')
    if mode =='mean' or mode =='0-1':
        cat_columns = cat_columns + ['outside_option']
    conti_index = generate_continuous_index(df_list[0], conti_columns)
    # print('finish generate_continuous_index', conti_index)
    cat_index = generate_category_index(df_list[0], cat_columns)
    # print('finish generate_category_index', cat_index)

    np_conti, np_cat, np_y, np_purchase, np_valid_len = generate_numpy_array(df_list, y_list, valid_len_list, total_list, conti_index, cat_index, max_num, mode)
    #print('finish generate_numpy_array')
    np.save('./True_Model/data/' + feature_size_list_name + ".npy", feature_sizes)
    np.save('./True_Model/data/' + data_name +'_conti.npy',np_conti)
    np.save('./True_Model/data/' + data_name +'_cat.npy',np_cat)
    if mode == "choice":
        np.save('./True_Model/data/' + choice_label_name +'.npy',np_y)
    elif mode == "purchase":
        np.save('./True_Model/data/' + purchase_label_name +'.npy',np_purchase)
    np.save('./True_Model/data/' + valid_length_name +'.npy',np_valid_len)
    # print(np_conti.shape)
    # print(np_cat.shape)
    # print(np_valid_len.shape)
    # print(np_purchase.shape)
    # print(np_y.shape)


def preprocess(csv_name, max_num, feature_size_list_name="category_feature_sizes", train_flag=True):

    # expedia dataset
    conti_columns = ['prop_location_score1', 'prop_location_score2', 
                    'prop_log_historical_price', 'price_usd', 'srch_adults_count', 'srch_children_count']
    # cat_columns = ['prop_id','prop_starrating', 'prop_review_score', 'prop_brand_bool','promotion_flag','srch_saturday_night_bool','srch_booking_window']
    cat_columns = ['prop_starrating', 'prop_review_score', 'prop_brand_bool','promotion_flag','srch_saturday_night_bool','srch_booking_window']
    # airline dataset
    # conti_columns = ["stayDurationMinutes", 'totalPrice', 'totalTripDurationMinutes', 'dtd', 'nAirlines', 
    #                 'nFlights', "outDepTime_sin", "outDepTime_cos", "outArrTime_sin", "outArrTime_cos"]
    # cat_columns = ['officeID', 'depWeekDay', 'OD', 'fAirline', 'staySaturday', 'isContinental', 'isDomestic']

    # expedia dataset
    # csv_name = 'df_OR_normal_process_with_nonpurchase.csv'
    orderid = 'srch_id'
    orderlabel = 'booking_bool'

    # airline dataset
    
    # orderid = 'orderid'
    # orderlabel = 'label'

    transfer_dataframe_into_npy(csv_name, orderid, orderlabel, conti_columns, cat_columns, max_num = max_num, data_name="data_ct", valid_length_name="valid_len_ct", feature_size_list_name = feature_size_list_name + "_ct", mode="choice", train_flag=train_flag)
    transfer_dataframe_into_npy(csv_name, orderid, orderlabel, conti_columns, cat_columns, max_num = max_num, data_name="data_pt", valid_length_name="valid_len_pt", feature_size_list_name = feature_size_list_name + "_pt",mode="purchase")

    # transfer_dataframe_into_npy(csv_name, orderid, orderlabel, conti_columns, cat_columns, "data_0_and_minus1", "label_0_and_minus1", "valid_len_0_and_minus1", "category_feature_sizes_0_and_minus1", mode="0-1")
