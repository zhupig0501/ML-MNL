import numpy as np
from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import os
import math


EMBEDDING_SIZE = 16
DROPOUT = 0.5
DEEPLAYER_SIZE = 64
FEATURE_SIZES = [6, 10, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2]
FEATURE_SIZES1 = np.array([6, 10, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1,2,2,2,2])
FIELD_SIZE = 12
k = 5

from sklearn.metrics import log_loss
#Define the loss function of MNL Model
class MNL_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y, temp_assortment_list):
        # batch_size,choice_size = x.shape
        batch_size = len(temp_assortment_list)
        loss = 0
        for i in range(batch_size):
            temp_sum = 0
            temp_y = 0
            # loss = loss + torch.log10((torch.exp(torch.sum(x[i]*y[i])))/(torch.sum(torch.exp(x[i]))))
            for j in temp_assortment_list[i]:
                temp_sum += torch.exp(x[j])
                if y[j] == 1:
                    temp_y = torch.exp(x[j])
            loss = loss + torch.log10(temp_y/(temp_sum))
        return -loss/batch_size

#Define the loss function of LC-MNL Model
class MMNL_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y,temp_assortment_list,z,alpha):
        batch_size = len(temp_assortment_list)
        model_num = alpha.shape[0]
        loss = 0
        for i in range(batch_size):
            g = 0
            for k in range(model_num):
                temp_sum = 0
                for j in temp_assortment_list[i]:
                    temp_sum += torch.exp(z[k,j])
                    if y[j] == 1:
                        temp_y = torch.exp(z[k,j])
                g = g + alpha[k]*temp_y/temp_sum
            g = float(g)
            loss = loss + torch.exp(torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]))/(torch.sum(torch.exp(x[temp_assortment_list[i]]))*g)
        return (-loss)/batch_size

#Define the loss function of Exponomial Choice Model
#Assume that no-purchase is 0
class Exp_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y,temp_assortment_list):
        batch_size = len(temp_assortment_list)
        loss = 0
        for i in range(batch_size):
            choice_size = len(temp_assortment_list[i])
            index_sort = torch.argsort(x[temp_assortment_list[i]])
            for j,item in enumerate(index_sort):
                if y[temp_assortment_list[i][item]] != 0:
                    if j == 0:
                        common_value = int(torch.sum(torch.max(x[temp_assortment_list[i]]-torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]),torch.tensor(0))))
                        loss = loss + torch.log((torch.exp(-torch.sum(torch.max(x[temp_assortment_list[i]]-torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]),torch.tensor(0)))+common_value)/(choice_size))) - common_value
                    else:
                        if j < 15:
                            common_value = int(torch.sum(torch.max(x[temp_assortment_list[i]]-torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]),torch.tensor(0))))
                        else:
                            common_value = int(torch.sum(torch.max(x[temp_assortment_list[i]]-x[temp_assortment_list[i][index_sort[j]]],torch.tensor(0))))
                        loss = loss + torch.log((torch.exp(-torch.sum(torch.max(x[temp_assortment_list[i]]-torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]),torch.tensor(0)))+common_value)/(choice_size- j) - torch.sum(torch.tensor([torch.exp(-torch.sum(torch.max(x[temp_assortment_list[i]]-x[temp_assortment_list[i][index_sort[k]]],torch.tensor(0)))+common_value)/((choice_size-k-1)*(choice_size-k)) for k in range(0,j)])))) - common_value
        return (-loss)/batch_size


#Define the MLE train model
class MLE(nn.Module):
     def __init__(self, field_size=FIELD_SIZE, feature_sizes=FEATURE_SIZES):
         super(MLE, self).__init__()
         self.field_size = field_size
         self.feature_sizes = feature_sizes
         self.bias = nn.Parameter(torch.randn(1))
         self.no_purchse = nn.Parameter(torch.randn(1))
         self.fm_first_order_embeddings = \
            nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in feature_sizes])

     def forward(self, X):
         X, weight = X
        #  batch_size,choice_size,feature_size = X.shape
         batch_size,feature_size = X.shape
        #  X = X.reshape(-1,feature_size)
        #  weight = weight.reshape(-1,feature_size)
         fm_first_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)]
         fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
         
        #  return (torch.sum(fm_first_order, 1)+self.bias).reshape(batch_size,choice_size),self.no_purchse
         return (torch.sum(fm_first_order, 1)+self.bias).reshape(batch_size)


#DeepFM Model
class DeepCell(nn.Module):
    def __init__(self, field_size = FIELD_SIZE, feature_sizes=FEATURE_SIZES,
                 embedding_size = EMBEDDING_SIZE, dropout = DROPOUT,
                 deeplayer_size = [32,64,32]):
        super(DeepCell, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.embedding_sizes = []
        # 计算embedding_sizes
        for i in range(len(feature_sizes)):
            if feature_sizes[i] != 1:
                self.embedding_sizes.append(math.ceil(k * math.log(feature_sizes[i])))
            else:
                self.embedding_sizes.append(1)
        self.dropout_shallow = [dropout, dropout]
        self.deep_layers = [i for i in deeplayer_size]
        self.dropout_deep = [dropout for i in range(len(deeplayer_size)+2)]
        self.bias = nn.Parameter(torch.randn(1))

        # FM Components
        self.fm_first_order_embeddings = \
            nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in feature_sizes])
        self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
        # self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, embedding_size) for feature_size in feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_sizes[i], self.embedding_sizes[i]) for i in range(len(feature_sizes))])
        self.fm_second_order_embedding_linear = nn.ModuleList([nn.Linear(self.embedding_sizes[i], embedding_size)
                                                               for i in range(len(self.embedding_sizes))])
        # self.fm_second_order_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_sizes[i], self.embedding_sizes[i]) for i in range(len(feature_sizes))])
        self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])

    

        # DNN Components
        self.linears = nn.ModuleList([nn.Linear(field_size * embedding_size, self.deep_layers[0])] +
                                     [nn.Linear(self.deep_layers[i], self.deep_layers[i + 1])
                                      for i in range(len(self.deep_layers) - 1)] + [nn.Linear(deeplayer_size[-1], 1)])
        self.liners_dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_deep])

    def forward(self, X):
        X, weight = X
        # FM Components
#         pdb.set_trace()
        batch_size,feature_size = X.shape
        X = X.reshape(-1,feature_size)
        weight = weight.reshape(-1,feature_size)

        fm_first_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_first_order = self.fm_first_order_dropout(fm_first_order)
#       print(weight[:, i].unsqueeze(-1).shape, emb(X[:, i]).shape)
#         pdb.set_trace()
#         fm_second_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr_linear = [lnr(fm_second_order_emb_arr[i]) for i, lnr in enumerate(self.fm_second_order_embedding_linear)]
        # fm_second_order_emb_sum = sum(fm_second_order_emb_arr)
        # fm_second_order_emb_sum_square = fm_second_order_emb_sum * fm_second_order_emb_sum
        # fm_second_order_emb_sum_square_sum = sum([item * item for item in fm_second_order_emb_arr])
        # fm_second_order = self.fm_second_order_dropout((fm_second_order_emb_sum_square -
        #                                                 fm_second_order_emb_sum_square_sum) * 0.5)
        fm_second_order_emb_sum = sum(fm_second_order_emb_arr_linear)
        fm_second_order_emb_sum_square = fm_second_order_emb_sum * fm_second_order_emb_sum
        fm_second_order_emb_sum_square_sum = sum([item * item for item in fm_second_order_emb_arr_linear])
        fm_second_order = self.fm_second_order_dropout((fm_second_order_emb_sum_square -
                                                        fm_second_order_emb_sum_square_sum) * 0.5)
       
        # DNN Components
        # deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_emb = torch.cat(fm_second_order_emb_arr_linear, 1)
        x_deep = self.liners_dropouts[0](deep_emb)

        for i in range(len(self.linears)):
            x_deep = self.linears[i](x_deep)
            x_deep = F.relu(x_deep)
            x_deep = self.liners_dropouts[i](x_deep)

        return torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + x_deep.reshape(-1) + self.bias

class DeepCell_With_Assortment(nn.Module):
    def __init__(self, field_size = FEATURE_SIZES1.shape[0], feature_sizes  =FEATURE_SIZES1,
                 embedding_size = EMBEDDING_SIZE, dropout = DROPOUT,
                 deeplayer_size = [32,64,32]):
        super(DeepCell_With_Assortment, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.embedding_sizes = []
        # 计算embedding_sizes
        for i in range(len(feature_sizes)):
            if feature_sizes[i] != 1:
                self.embedding_sizes.append(math.ceil(k * math.log(feature_sizes[i])))
            else:
                self.embedding_sizes.append(1)
        self.dropout_shallow = [dropout, dropout]
        self.deep_layers = [i for i in deeplayer_size]
        self.dropout_deep = [dropout for i in range(len(deeplayer_size)+2)]
        self.bias = nn.Parameter(torch.randn(1))

        # FM Components
        self.fm_first_order_embeddings = \
            nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in feature_sizes])
        self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
        # self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, embedding_size) for feature_size in feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_sizes[i], self.embedding_sizes[i]) for i in range(len(feature_sizes))])
        self.fm_second_order_embedding_linear = nn.ModuleList([nn.Linear(self.embedding_sizes[i], embedding_size)
                                                               for i in range(len(self.embedding_sizes))])
        # self.fm_second_order_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_sizes[i], self.embedding_sizes[i]) for i in range(len(feature_sizes))])
        self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])

    

        # DNN Components
        self.linears = nn.ModuleList([nn.Linear(field_size * embedding_size, self.deep_layers[0])] +
                                     [nn.Linear(self.deep_layers[i], self.deep_layers[i + 1])
                                      for i in range(len(self.deep_layers) - 1)] + [nn.Linear(deeplayer_size[-1], 1)])
        self.liners_dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_deep])

    def forward(self, X):
        X, weight = X
        # FM Components
#         pdb.set_trace()
        batch_size,feature_size = X.shape
        X = X.reshape(-1,feature_size)
        weight = weight.reshape(-1,feature_size)

        fm_first_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_first_order = self.fm_first_order_dropout(fm_first_order)
#       print(weight[:, i].unsqueeze(-1).shape, emb(X[:, i]).shape)
#         pdb.set_trace()
#         fm_second_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = [weight[:, i].unsqueeze(-1) * emb(X[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr_linear = [lnr(fm_second_order_emb_arr[i]) for i, lnr in enumerate(self.fm_second_order_embedding_linear)]
        # fm_second_order_emb_sum = sum(fm_second_order_emb_arr)
        # fm_second_order_emb_sum_square = fm_second_order_emb_sum * fm_second_order_emb_sum
        # fm_second_order_emb_sum_square_sum = sum([item * item for item in fm_second_order_emb_arr])
        # fm_second_order = self.fm_second_order_dropout((fm_second_order_emb_sum_square -
        #                                                 fm_second_order_emb_sum_square_sum) * 0.5)
        fm_second_order_emb_sum = sum(fm_second_order_emb_arr_linear)
        fm_second_order_emb_sum_square = fm_second_order_emb_sum * fm_second_order_emb_sum
        fm_second_order_emb_sum_square_sum = sum([item * item for item in fm_second_order_emb_arr_linear])
        fm_second_order = self.fm_second_order_dropout((fm_second_order_emb_sum_square -
                                                        fm_second_order_emb_sum_square_sum) * 0.5)
       
        # DNN Components
        # deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_emb = torch.cat(fm_second_order_emb_arr_linear, 1)
        x_deep = self.liners_dropouts[0](deep_emb)

        for i in range(len(self.linears)):
            x_deep = self.linears[i](x_deep)
            x_deep = F.relu(x_deep)
            x_deep = self.liners_dropouts[i](x_deep)

        return torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + x_deep.reshape(-1) + self.bias

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.save_path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss