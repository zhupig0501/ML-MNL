import numpy as np
import pandas as pd
import Part_2_dataframe_preprocess
import ChoiceTfData
import ChoiceTransformer
import PurchaseTransformer

def train_model(df,pt_weight = [6,1],model_name = None):
    purchase_rate = np.array([np.sum(df['booking_bool'])/len(list(df['srch_id'].unique()))])
    np.save(".\\Expedia_Transformer\\data\\purchase_rate_pt_" + model_name +  ".npy",purchase_rate)
    Part_2_dataframe_preprocess.preprocess(df, max_num=None)

    ctd_ct = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                    data_cat_name="data_ct_cat", data_conti_name="data_ct_conti", 
                                    valid_len_name="valid_len_ct", model_path="ct", purchase_flag=False)
    ctd_ct.generate_train_valid_test_data()
    ctd_pt = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                        data_cat_name="data_pt_cat", data_conti_name="data_pt_conti", 
                                        valid_len_name="valid_len_pt", model_path="pt", purchase_flag=True)
    ctd_pt.generate_train_valid_test_data()
    pt_model_path = "pt_" + model_name
    ct_model_path = "ct_" + model_name
    # for 200, weight = [6,1] for 50, weight = [2.3, 1]
    # The following doesn't need to be changed
    batch_path_ct = "./Expedia_Transformer/data/batch_size_ct_" + ct_model_path + ".npy"
    batch_path_pt = "./Expedia_Transformer/data/batch_size_pt_" + pt_model_path + ".npy"

    # train purchase model
    pt = PurchaseTransformer.PurchaseTransformer(weight=pt_weight)
    pt.load_data(ctd_pt)
    pt.train_model(pt_model_path)

    # train choice model
    ct = ChoiceTransformer.ChoiceTransformer()
    ct.load_data(ctd_ct)
    ct.train_model(ct_model_path)

def predict(df, max_num = 38 , pt_weight = [6,1],model_name = "true_model"):
    pt_model_path = "pt_" + model_name
    ct_model_path = "ct_" + model_name
    # for 200, weight = [6,1] for 50, weight = [2.3, 1]
    # The following doesn't need to be changed
    batch_path_ct = "./Expedia_Transformer/data/batch_size_ct_" + ct_model_path + ".npy"
    batch_path_pt = "./Expedia_Transformer/data/batch_size_pt_" + pt_model_path + ".npy"
    #csv_name = 'E:\\OneDrive - sjtu.edu.cn\\ML-mnl-new\\train_dataset_for_transformer\\200_20000.csv'
    Part_2_dataframe_preprocess.preprocess(df, feature_size_list_name = "category_feature_sizes_new", max_num = max_num, train_flag=False)

    ctd_ct_test = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                    data_cat_name="data_ct_cat", data_conti_name="data_ct_conti", 
                                    valid_len_name="valid_len_ct", model_path="ct", purchase_flag=False)
    ct = ChoiceTransformer.ChoiceTransformer()
    ct.load_data(ctd_ct_test, train_flag=False)
    ct.initialize_model(model_path=ct_model_path, batch_path = batch_path_ct)
    ct_choice = ct.get_the_choice_rate()

    ctd_pt_test = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                    data_cat_name="data_pt_cat", data_conti_name="data_pt_conti", 
                                    valid_len_name="valid_len_pt", model_path="pt", purchase_flag=True)
    pt = PurchaseTransformer.PurchaseTransformer(weight=pt_weight)
    pt.load_data(ctd_pt_test, train_flag=False)
    pt.initialize_model(model_path=pt_model_path, batch_path = batch_path_pt)
    pt_purchase = pt.get_the_purchase_rate()

    # get the final result
    final_result = ct_choice * pt_purchase.unsqueeze(1)
    return final_result.reshape(final_result.shape[0],final_result.shape[1])
