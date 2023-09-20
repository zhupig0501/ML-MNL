import ChoiceTfData
import ChoiceTransformer
import PurchaseTransformer

ctd_ct = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_ct", 
                                    data_cat_name="data_ct_cat", data_conti_name="data_ct_conti", 
                                    valid_len_name="valid_len_ct", model_path="ct", purchase_flag=False)
ctd_ct.generate_train_valid_test_data()
ctd_pt = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                    data_cat_name="data_pt_cat", data_conti_name="data_pt_conti", 
                                    valid_len_name="valid_len_pt", model_path="pt", purchase_flag=True)
ctd_pt.generate_train_valid_test_data()

pt_model_path = "pt_200_20000"
ct_model_path = "ct_200_20000"
batch_path_ct = "./data/batch_size_ct_" + ct_model_path + ".npy"
batch_path_pt = "./data/batch_size_pt_" + pt_model_path + ".npy"
# for 200, weight = [6,1] for 50, weight = [2.3, 1]
pt_weight = [6,1]

# train models

# train purchase model
pt = PurchaseTransformer.PurchaseTransformer(weight=pt_weight)
pt.load_data(ctd_pt)
pt.train_model(pt_model_path)

# train choice model
ct = ChoiceTransformer.ChoiceTransformer()
ct.load_data(ctd_ct)
ct.train_model(ct_model_path)

# get different rate and result
# get choice rate
ctd_ct_test = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_ct", 
                                    data_cat_name="test_data_cat_ct", data_conti_name="test_data_conti_ct", 
                                    valid_len_name="test_valid_len_ct", model_path="ct", purchase_flag=False)
ct = ChoiceTransformer.ChoiceTransformer()
ct.load_data(ctd_ct_test, train_flag=False)
ct.initialize_model(model_path=ct_model_path, batch_path = batch_path_ct)
ct_choice = ct.get_the_choice_rate()
# print(ct_choice.shape)

# get purchase rate
ctd_pt_test = ChoiceTfData.ChoiceTfData(feature_size_list_name="category_feature_sizes_pt", 
                                    data_cat_name="test_data_cat_pt", data_conti_name="test_data_conti_pt", 
                                    valid_len_name="test_valid_len_pt", model_path="pt", purchase_flag=True)
pt = PurchaseTransformer.PurchaseTransformer(weight=pt_weight)
pt.load_data(ctd_pt_test, train_flag=False)
pt.initialize_model(model_path=pt_model_path, batch_path = batch_path_pt)
pt_purchase = pt.get_the_purchase_rate()
# print(pt_purchase.shape)

# get the final result
final_result = ct_choice * pt_purchase.unsqueeze(1)

