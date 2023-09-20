import numpy as np
class ChoiceTfData():
    def __init__(self, feature_size_list_name=None, data_cat_name=None, data_conti_name=None, purchase_label_name=None, choice_label_name=None, valid_len_name=None, model_path = None, purchase_flag = True):
        # load all dataset generate from part2 dataframe preprocess
        self.model_path = model_path
        self.purchase_flag = purchase_flag
        if feature_size_list_name is None:
            self.feature_size_list = list(np.load("./True_Model/data/category_feature_sizes.npy"))
        else:
            self.feature_size_list = list(np.load("./True_Model/data/"+feature_size_list_name + ".npy"))
            #print(self.feature_size_list)
        if data_cat_name is None:
            self.data_cat = np.load("./True_Model/data/data_cat.npy")
        else:
            self.data_cat = np.load("./True_Model/data/" + data_cat_name + ".npy")
        if data_conti_name is None:
            self.data_conti = np.load("./True_Model/data/data_conti.npy")
        else:
            self.data_conti = np.load("./True_Model/data/" + data_conti_name + ".npy")
        if purchase_flag:
            if purchase_label_name is None:
                self.purchase_label = np.load("./True_Model/data/purchase_label.npy")
            else:
                self.purchase_label = np.load("./True_Model/data/" + purchase_label_name + ".npy")
        else:
            if choice_label_name is None:
                self.choice_label = np.load("./True_Model/data/choice_label.npy")
            else:
                self.choice_label = np.load("./True_Model/data/" + choice_label_name + ".npy")
        if valid_len_name is None:
            self.valid_len = np.load("./True_Model/data/valid_len.npy")
        else:
            self.valid_len = np.load("./True_Model/data/" + valid_len_name + ".npy")
        #print("load all dataset generate from part2 dataframe preprocess successfully")

        
    def generate_train_valid_test_data(self, train_flag=True):
        # generate train valid test data
        total = self.data_conti.shape[0]
        if self.purchase_flag:
            if train_flag:
                self.purchase_rate = self.purchase_label.sum() / self.purchase_label.shape[0]
                np.save(".\\True_Model\\data\\purchase_rate" +self.model_path +  ".npy",self.purchase_rate)
            # first pick up the data which customer do not choose any product
            non_purchase_indices = []
            for i in range(total):
                if sum(self.purchase_label[i]) == 0:
                    non_purchase_indices.append(i)
            non_purchase_indices = np.array(non_purchase_indices)
            # then get the indices for the data which customer choose one product
            array = np.arange(0, total)
            if len(non_purchase_indices) > 0 :
                purchase_indices = np.delete(array, non_purchase_indices)
            else:
                purchase_indices = array

            # shuffle the data
            non_purchase_indices = np.random.permutation(non_purchase_indices)
            purchase_indices = np.random.permutation(purchase_indices)
            
            self.data_cat_ct = self.data_cat[purchase_indices]
            self.data_conti_ct = self.data_conti[purchase_indices]


            non_purchase_num = non_purchase_indices.shape[0]
            purchase_num = purchase_indices.shape[0]
            # split train, valid, test data with the ratio 8:1:1
            # get training length for two kinds of data
            non_purchase_training_len = int(non_purchase_num * 0.8)
            purchase_training_len = int(purchase_num * 0.8)
            # get valid length for two kinds of data
            non_purchase_valid_len = int(non_purchase_num * 0.1)
            purchase_valid_len = int(purchase_num * 0.1)
            # get test length for two kinds of data
            non_purchase_test_len = non_purchase_num - non_purchase_training_len - non_purchase_valid_len
            purchase_test_len = purchase_num - purchase_training_len - purchase_valid_len


            # note that choice transformer can only use purchase data 
            # purchase transformer can use both purchase and non-purchase data
            # get indices for training, valid, test data
            train_indices = np.concatenate((non_purchase_indices[0:non_purchase_training_len], purchase_indices[0:purchase_training_len]))
            valid_indices = np.concatenate((non_purchase_indices[non_purchase_training_len:non_purchase_training_len + non_purchase_valid_len], purchase_indices[purchase_training_len:purchase_training_len + purchase_valid_len]))
            test_indices = np.concatenate((non_purchase_indices[non_purchase_training_len + non_purchase_valid_len:], purchase_indices[purchase_training_len + purchase_valid_len:]))
            train_indices = np.random.permutation(train_indices)
            valid_indices = np.random.permutation(valid_indices)
            test_indices = np.random.permutation(test_indices)
            # get training, valid, test length
            self.training_length = non_purchase_training_len + purchase_training_len
            self.valid_length = non_purchase_valid_len + purchase_valid_len
            self.test_length = non_purchase_test_len + purchase_test_len

            # get training, valid, test data
        else:
            array = np.arange(0, total)
            purchase_indices = np.random.permutation(array)
            purchase_num = purchase_indices.shape[0]
            # split train, valid, test data with the ratio 8:1:1
            purchase_training_len = int(purchase_num * 0.8)
            purchase_valid_len = int(purchase_num * 0.1)
            purchase_test_len = purchase_num - purchase_training_len - purchase_valid_len
            
            train_indices = purchase_indices[0:purchase_training_len]
            valid_indices = purchase_indices[purchase_training_len:purchase_training_len + purchase_valid_len]
            test_indices = purchase_indices[purchase_training_len + purchase_valid_len:]
            # get training, valid, test length
            self.training_length  = purchase_training_len
            self.valid_length  = purchase_valid_len
            self.test_length = total - self.training_length - self.valid_length

        # generate dataset for purchase transformer
        if self.purchase_flag:
            label = self.purchase_label
        else:
            label = self.choice_label
        self.training_data = [self.data_conti[train_indices], self.data_cat[train_indices], label[train_indices], self.valid_len[train_indices]]
        self.valid_data = [self.data_conti[valid_indices], self.data_cat[valid_indices], label[valid_indices], self.valid_len[valid_indices]]
        self.test_data = [self.data_conti[test_indices], self.data_cat[test_indices], label[test_indices], self.valid_len[test_indices]]
   
        if self.model_path is not None:
            # test data
            # save data for purchase_tranformer
            np.save(".\\True_Model\\data\\test_data_conti_" +self.model_path +  ".npy",self.data_conti[test_indices])
            np.save(".\\True_Model\\data\\test_data_cat_" +self.model_path +  ".npy", self.data_cat[test_indices])
            if self.purchase_flag:
                np.save(".\\True_Model\\data\\test_purchase_label_" +self.model_path +  ".npy", self.purchase_label[test_indices])
            else:
                np.save(".\\True_Model\\data\\test_choice_label_" +self.model_path +  ".npy", self.choice_label[test_indices])
            np.save(".\\True_Model\\data\\test_valid_len_" +self.model_path +  ".npy", self.valid_len[test_indices])
            # save data for choice transformer
   
        else:
            # test data
            # save data for purchase_tranformer
            np.save(".\\True_Model\\data\\test_data_conti.npy",self.data_conti[test_indices])
            np.save(".\\True_Model\\data\\test_data_cat.npy", self.data_cat[test_indices])
            if self.purchase_flag:
                np.save(".\\True_Model\\data\\test_purchase_label.npy", self.purchase_label[test_indices])
            else:
                np.save(".\\True_Model\\data\\test_choice_label.npy", self.choice_label[test_indices])
            np.save(".\\True_Model\\data\\test_valid_len.npy", self.valid_len[test_indices])
            # save data for choice transformer
            #print("generate_train_valid_test_data successfully")
