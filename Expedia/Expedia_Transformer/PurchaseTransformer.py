import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
import torch.nn as nn
from d2l import torch as d2l
from True_Model.Transformer_nonpurchase import PurchaseTransformerEncoder
from True_Model.Transformer_nonpurchase import EarlyStopping
from True_Model.Transformer_nonpurchase import CELoss
import math
from tqdm import tqdm
import numpy as np

def weight_constraint(weights):
    return torch.relu(weights)

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def min_divisor_geq_4(n):
    for i in range(4, n + 1):
        if n % i == 0:
            return i
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    return 1

class PurchaseTransformer():
    def __init__(self, non_purchase_flag=True, weight=[30,1]):
        # setting save path in current directory
        self.save_path = ".\\True_Model\\checkpoints\\" 
        
        # weight for solving unbalanced data in assortment
        self.weight = torch.from_numpy(np.array(weight)).float()
        
        # setting device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        #print(self.device)

        # set criterion loss for choice transformer
        self.criterion = CELoss(weight=self.weight)

        self.non_purchase_flag = non_purchase_flag



    def generate_train_valid_test_batch(self):

        # generate valid test batch
        self.valid_batches, self.test_batches = 0, 0
        self.generate_valid_test_batch()
        #print("generate valid test batch successfully")

    def train_model(self, model_path=None):
        self.set_hyperparameter_and_model_path(model_path=model_path)
        self.generate_train_valid_test_batch()
        # train model
        self.train_choice_transformer()
        # load model
        # self.load_model()
        # # test model
        # self.test_purchase_transformer()

    def initialize_model(self, batch_path=None, model_path=None):
        if batch_path is None:
            batch_path = "./True_Model/data/batch_size_pt.npy"
        batch_size = np.load(batch_path)
        self.set_hyperparameter_and_model_path(batch_size, model_path=model_path)
        self.generate_data_batch()
        self.load_model()

        
    def load_data(self, ChoiceTfData, train_flag=True):
        # load all dataset generate from part2.5 ChoiceTfData
        if train_flag:
            self.training_data = ChoiceTfData.training_data
            self.valid_data = ChoiceTfData.valid_data
            self.test_data = ChoiceTfData.test_data
            self.training_length = ChoiceTfData.training_length
            self.valid_length = ChoiceTfData.valid_length
            self.test_length = ChoiceTfData.test_length

        self.data_conti = ChoiceTfData.data_conti
        self.data_cat = ChoiceTfData.data_cat
        self.feature_size_list = ChoiceTfData.feature_size_list
        self.valid_len = ChoiceTfData.valid_len
        self.purchase_label = ChoiceTfData.purchase_label

        #print("load all dataset generate from part2.5 generate_train_valid_test_data successfully")
    
    def set_hyperparameter_and_model_path(self,batch_size=None, model_path=None):
        # set hyperparameter
        self.hp = hyper_parameter(self.data_conti, self.data_cat, self.feature_size_list, batch_size)
        self.hp.set_num_layers(1)
        self.hp.set_dropout_rate(0.5)
        self.early_stopping = EarlyStopping(self.save_path, patience=self.hp.patience, model_path=model_path)
        self.model_path = model_path
        if self.model_path is None:
            np.save(".\\True_Model\\data\\batch_size_pt.npy", self.hp.batch_size)
        else:
            np.save(".\\True_Model\\data\\batch_size_pt_" +self.model_path +  ".npy", self.hp.batch_size)
        #print("set hyperparameter successfully")
        # hp.print_all_hyperparameters()

    def train_choice_transformer(self):   
        # train choice transformer
        self.model = PurchaseTransformerEncoder(
            self.hp.key_size, self.hp.query_size, self.hp.value_size, self.hp.num_hiddens,
            self.hp.norm_shape, self.hp.ffn_num_input, self.hp.ffn_num_hiddens, self.hp.num_heads,
            self.hp.num_layers, self.hp.dropout, self.feature_size_list, self.hp.k, self.hp.embedding_sizes, self.hp.norm_shape_init, self.device, self.data_conti.shape[1])


        self.model.apply(xavier_init_weights)
        self.model = self.model.to(self.device)
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.hp.lr)


        for epoch in range(self.hp.num_epochs):
            print(f'---------------- Epoch: {epoch+1:02} ----------------')
            # print("lr:", optimizer.get_lr())
            indices = np.random.permutation(self.training_length)
            training_data_conti, training_data_cat, training_label, training_valid_len = self.training_data
            train_data = [training_data_cat[indices], training_data_conti[indices], training_label[indices], training_valid_len[indices]]
            train_batches = list()
            i = 0
            while i < self.training_length:
                j = min(i + self.hp.batch_size, self.training_length)
                train_batches.append([train_data[0][i: j], train_data[1][i: j], train_data[2][i: j], train_data[3][i:j]])
                i = j
            loss_total = 0
            length_total = 0
            # for step, batch in enumerate(tqdm(train_batches)):
            for step, batch in enumerate(train_batches):
                self.model.train()
                optimizer.zero_grad()
        #         pdb.set_trace()
                cat = torch.from_numpy(batch[0]).to(torch.long).to(self.device)
                conti = torch.from_numpy(batch[1]).to(torch.float).to(self.device)
                valid_lengt = torch.from_numpy(batch[3]).to(torch.long).to(self.device)
                outputs = self.model([cat, conti, valid_lengt]).to(self.device)
                y = torch.from_numpy(batch[2]).to(torch.long).to(self.device)
                loss = self.criterion(outputs, y, valid_lengt)
                loss_total += float(loss)
                length_total += float(sum(valid_lengt))
                loss.backward()
                d2l.grad_clipping(self.model, 8)
                optimizer.step()
                # self.model.dense2.weight.data = weight_constraint(self.model.dense2.weight.data)
            print('Epoch', epoch, 'loss', loss_total/ length_total)
        #         if step % DISPLAY == 0:
        #     scheduler.step()
            with torch.no_grad():
                self.model.eval()
                loss_total = 0
                length_total = 0
                purchase_acc = 0
                non_purchase_acc = 0
                purchase_count = 0
                non_purchase_count = 0
                acc = 0
                for step, batch in enumerate(self.valid_batches):
                    cat = torch.from_numpy(batch[0]).to(torch.long).to(self.device)
                    conti = torch.from_numpy(batch[1]).to(torch.float).to(self.device)
                    valid_lens = torch.from_numpy(batch[3]).to(torch.long).to(self.device)
                    outputs = self.model([cat, conti, valid_lens])
                    y = torch.from_numpy(batch[2]).to(torch.long).to(self.device)
                    loss_total += self.criterion(outputs, y, valid_lens)
                    length_total += float(sum(valid_lens))
                    j = outputs.shape[0]
                print("valid loss {:g}".format(loss_total / length_total))
                if epoch >= self.hp.warmup_steps:
                    self.early_stopping(loss_total, self.model)
                    # if early stopping is triggered, break the loop
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break  
    
    def generate_valid_test_batch(self):
        # generate valid and test batch
        self.valid_batches = list()
        indices = np.random.permutation(self.valid_length)
        valid_data_conti, valid_data_cat, valid_label, valid_valid_len = self.valid_data
        valid_data = [valid_data_cat[indices], valid_data_conti[indices], valid_label[indices], valid_valid_len[indices]]
        i = 0
        while i < self.valid_length:
            j = min(i + self.hp.batch_size, self.valid_length)
            self.valid_batches.append([valid_data[0][i: j], valid_data[1][i: j], valid_data[2][i: j], valid_data[3][i:j]])
            i = j

        self.test_batches = list()
        indices = np.random.permutation(self.test_length)
        test_data_conti, test_data_cat, test_label, test_valid_len = self.test_data
        test_data = [test_data_cat[indices], test_data_conti[indices], test_label[indices],
                    test_valid_len[indices]]
        i = 0
        while i < self.test_length:
            j = min(i + self.hp.batch_size, self.test_length)
            self.test_batches.append([test_data[0][i: j], test_data[1][i: j], test_data[2][i: j], test_data[3][i:j]])
            i = j

    def generate_data_batch(self):
        self.test_batches = list()
        self.test_length = self.data_conti.shape[0]
        # indices = np.random.permutation(self.test_length)
        test_data_conti, test_data_cat, test_label, test_valid_len = self.data_conti, self.data_cat, self.purchase_label, self.valid_len
        # test_data = [test_data_cat[indices], test_data_conti[indices], test_label[indices],
        #             test_valid_len[indices]]
        test_data = [test_data_cat, test_data_conti, test_label, test_valid_len]
        i = 0
        while i < self.test_length:
            j = min(i + self.hp.batch_size, self.test_length)
            self.test_batches.append([test_data[0][i: j], test_data[1][i: j], test_data[2][i: j], test_data[3][i:j]])
            i = j

    def load_model(self):
        torch.cuda.empty_cache()
        self.model = PurchaseTransformerEncoder(
            self.hp.key_size, self.hp.query_size, self.hp.value_size, self.hp.num_hiddens,
            self.hp.norm_shape, self.hp.ffn_num_input, self.hp.ffn_num_hiddens, self.hp.num_heads,
            self.hp.num_layers, self.hp.dropout, self.feature_size_list, self.hp.k, self.hp.embedding_sizes, self.hp.norm_shape_init, self.device, self.data_conti.shape[1])
        if self.model_path is None:
            model_path = ".\\True_Model\\checkpoints\\best_network.pth"
        else:
            model_path = ".\\True_Model\\checkpoints\\" + self.model_path + ".pth"
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)

    def test_purchase_transformer(self):
        with torch.no_grad():
            self.model.eval()
            loss_total = 0
            acc = 0
            length_total = 0
            for step, batch in enumerate(self.test_batches):
                cat = torch.from_numpy(batch[0]).to(torch.long).to(self.device)
                conti = torch.from_numpy(batch[1]).to(torch.float).to(self.device)
                valid_lens = torch.from_numpy(batch[3]).to(torch.long).to(self.device)
                outputs = self.model([cat, conti, valid_lens])
                y = torch.from_numpy(batch[2]).to(torch.long).to(self.device)
                loss = self.criterion(outputs, y, valid_lens)
                loss_total += loss
                length_total += float(sum(valid_lens))
            print("test loss {:g}".format(np.sqrt(loss_total / length_total)))

    def get_the_purchase_rate(self):
        # purchase_rate = np.load(".\\True_Model\\data\\purchase_rate_" +self.model_path +  ".npy")
        cat = torch.from_numpy(self.data_cat).to(torch.long).to(self.device)
        conti = torch.from_numpy(self.data_conti).to(torch.float).to(self.device)
        valid_lens = torch.from_numpy(self.valid_len).to(torch.long).to(self.device)
        outputs = self.model([cat, conti, valid_lens])
        # mean_num = torch.mean(outputs)
        valid_ratio = ((valid_lens/self.data_cat.shape[1]).reshape([self.data_cat.shape[0], 1]))
        outputs = outputs * valid_ratio
        return (outputs)

class hyper_parameter():
    def __init__(self, data_conti, data_cat, feature_size_list, batch_size):
        # for each feature i: Di = log(Feature_sizes) * k
        self.k = 5
        if batch_size == None:
            self.batch_size, self.choice_size, self.feature_size = int(data_conti.shape[0] * 0.005) , data_conti.shape[1], data_conti.shape[2] + data_cat.shape[2]
        else:
            self.batch_size, self.choice_size, self.feature_size = batch_size,  data_conti.shape[1], data_conti.shape[2] + data_cat.shape[2]
        # [D1, D2, D3, D4, D5, D6, ..., Dn]
        self.embedding_sizes = []
        self.calculate_embedding_sizes(feature_size_list)
        #print("embedding_sizes", self.embedding_sizes)
        # self_attention key,query,value size, the final hidden layer size
        self.key_size=self.query_size=self.value_size = self.num_hiddens = sum(self.embedding_sizes) + data_conti.shape[2]
        # the final hidden layer size, the number of encoder layers, dropout rate
        self.num_layers, self.dropout = 0,0
        # layer normalization shape
        self.norm_shape = [self.choice_size, self.key_size]
        # norm_shape_init(The number of continuous features)
        self.norm_shape_init = data_conti.shape[2]
        # FFN input num, FFN hidden num, multi-head attention num
        self.ffn_num_input, self.ffn_num_hiddens, self.num_heads = self.key_size, 2*self.key_size , min_divisor_geq_4(self.key_size)

        # epoch num, learning rate, warmup_steps
        self.num_epochs, self.lr, self.warmup_steps = 100, 0.3, 5
        # patience related with early stop
        self.patience = 3

    def set_num_layers(self, num_layers):
        self.num_layers = num_layers
    
    def set_dropout_rate(self, dropout):
        self.dropout = dropout

    def calculate_embedding_sizes(self, feature_size_list):
        for i in range(len(feature_size_list)):
            if feature_size_list[i] != 1:
                self.embedding_sizes.append(math.ceil(self.k * math.log(feature_size_list[i])))
            else:
                self.embedding_sizes.append(1)
    
    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs
    
    def set_learning_rate(self, lr):
        self.lr = lr
    
    def set_warmup_steps(self, warmup_steps):
        self.warmup_steps = warmup_steps
    
    def set_norm_shape_init(self, norm_shape_init):
        self.norm_shape_init = norm_shape_init

    def print_all_hyperparameters(self):
        print('-------------------------------------------------------------------------------')
        print("k is: ", self.k)
        print("batch_size, choice_size, feature_size is: ", self.batch_size, self.choice_size, self.feature_size)
        print("embedding_sizes is: ", self.embedding_sizes)
        print("key_size, query_size, value_size, num_hiddens is: ", self.key_size, self.query_size, self.value_size, self.num_hiddens)
        print("num_layers, dropout is: ", self.num_layers, self.dropout)
        print("norm_shape is: ", self.norm_shape)
        print("norm_shape_init is: ", self.norm_shape_init)
        print("ffn_num_input, ffn_num_hiddens, num_heads is: ", self.ffn_num_input, self.ffn_num_hiddens, self.num_heads)
        print("num_epochs, lr, warmup_steps is: ", self.num_epochs, self.lr, self.warmup_steps)
        print("patience is: ", self.patience)
        print('-------------------------------------------------------------------------------')
        

# 50-20000 weight=2.3, 200-20000 weight =6
# purchase_tf = PurchaseTransformer(weight=[2.3,1])
# for mode = mean
# purchase_tf.load_data()
# for mode = 0-1
# purchase_tf.load_data('category_feature_sizes_0_and_minus1', 'data_0_and_minus1_cat', 'data_0_and_minus1_conti', 'label_0_and_minus1', 'valid_len_0_and_minus1')
# purchase_tf.train_model()


# purchase_tf = PurchaseTransformer()
# # choice_tf.load_data(data_cat_name="test_data_cat", data_conti_name="test_data_conti", label_name="test_label", valid_len_name="test_valid_len")
# purchase_tf.load_data(data_cat_name="test_data_cat_15", data_conti_name="test_data_conti_15", label_name="test_label", valid_len_name="test_valid_len_15")

# purchase_tf.initialize_model()
# data = [purchase_tf.data_cat[0:10], purchase_tf.data_conti[0:10], purchase_tf.valid_len[0:10]]
# purchase_rate = purchase_tf.get_the_purchase_rate(data)
# print("purchase_rate", purchase_rate)