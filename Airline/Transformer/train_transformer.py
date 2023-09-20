import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
import torch.nn as nn
from d2l import torch as d2l
from Transformer import TransformerEncoder
from Transformer import EarlyStopping
import math
from Transformer import MaskedSoftmaxCELoss
from tqdm import tqdm
import numpy as np


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


# all parameter used to train Transformer
feature_sizes = [11, 7, 97, 63, 2, 2, 2]
k = 5
batch_size, choice_size, feature_size = 128, 320, 17
key_size = query_size = value_size = 88
num_hiddens, num_layers, dropout = 88, 1, 0.5
norm_shape = [320, 88]
norm_shape_init = 10
ffn_num_input, ffn_num_hiddens, num_heads = 88, 176, 4
num_epochs = 100
lr = 0.3
warmup_steps = 5
# 提前停止的步数
early_stop_num = 5
weight = torch.from_numpy(np.array([30.0])).float()
patience = 10

embedding_sizes = []
# 计算embedding_sizes
for i in range(len(feature_sizes)):
    if feature_sizes[i] != 1:
        embedding_sizes.append(math.ceil(k * math.log(feature_sizes[i])))
    else:
        embedding_sizes.append(1)

# data_loading
data_conti = np.load("data_conti_trans320.npy")
data_cat = np.load("data_cat_trans320.npy")
label = np.load("label_trans320.npy")
valid_len = np.load("valid_len_trans320.npy")

total = data_conti.shape[0]
indices = np.random.permutation(total)
data_conti = data_conti[indices]
data_cat = data_cat[indices]
valid_len = valid_len[indices]
label = label[indices]

save_path = ".\\checkpoints\\"
early_stopping = EarlyStopping(save_path, patience=patience)

# 将数据按照8：1：1的比例分成三块
training_length = int(total*0.8)
valid_length = int(total*0.1)
test_length = total - training_length - valid_length

training_data_conti = data_conti[0:training_length]
training_data_cat = data_cat[0:training_length]
training_label = label[0:training_length]
training_valid_len = valid_len[0:training_length]


valid_data_conti = data_conti[training_length:training_length+valid_length]
valid_data_cat = data_cat[training_length:training_length+valid_length]
valid_label = label[training_length:training_length+valid_length]
valid_valid_len = valid_len[training_length:training_length+valid_length]

test_data_conti = data_conti[training_length+valid_length:]
test_data_cat = data_cat[training_length+valid_length:]
test_label = label[training_length+valid_length:]
test_valid_len = valid_len[training_length+valid_length:]

np.save(".\\np_data\\test_data_conti.npy", test_data_conti)
np.save(".\\np_data\\test_data_cat.npy", test_data_cat)
np.save(".\\np_data\\test_label.npy", test_label)
np.save(".\\np_data\\test_valid_len.npy", test_valid_len)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TransformerEncoder(
    key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, feature_sizes, k, embedding_sizes, norm_shape_init)

model.apply(xavier_init_weights)
model = model.to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)


criterion = MaskedSoftmaxCELoss(weight).to(device)
old_acc = 0
i = 0
valid_batches = list()
indices = np.random.permutation(valid_length)
valid_data = [valid_data_cat[indices], valid_data_conti[indices], valid_label[indices], valid_valid_len[indices]]
while i < valid_length:
    j = min(i + batch_size, valid_length)
    valid_batches.append([valid_data[0][i: j], valid_data[1][i: j], valid_data[2][i: j], valid_data[3][i:j]])
    i = j

test_batches = list()
indices = np.random.permutation(test_length)
test_data = [test_data_cat[indices], test_data_conti[indices], test_label[indices],
             test_valid_len[indices]]
i = 0
while i < test_length:
    j = min(i + batch_size, test_length)
    test_batches.append([test_data[0][i: j], test_data[1][i: j], test_data[2][i: j], test_data[3][i:j]])
    i = j

for epoch in range(num_epochs):
    print(f'---------------- Epoch: {epoch+1:02} ----------------')
    # print("lr:", optimizer.get_lr())
    indices = np.random.permutation(training_length)
    train_data = [training_data_cat[indices], training_data_conti[indices], training_label[indices], training_valid_len[indices]]
    train_batches = list()
    i = 0
    while i < training_length:
        j = min(i + batch_size, training_length)
        train_batches.append([train_data[0][i: j], train_data[1][i: j], train_data[2][i: j], train_data[3][i:j]])
        i = j
    loss_total = 0
    for step, batch in enumerate(tqdm(train_batches)):
        model.train()
        optimizer.zero_grad()
        cat = torch.from_numpy(batch[0]).to(torch.long).to(device)
        conti = torch.from_numpy(batch[1]).to(torch.float).to(device)
        valid_lengt = torch.from_numpy(batch[3]).to(torch.long).to(device)
        outputs = model([cat, conti, valid_lengt]).to(device)
        y = torch.from_numpy(batch[2]).to(torch.long).to(device)
        loss = criterion(outputs, y, valid_lengt)
        loss_total += float(loss)
        loss.backward()
        d2l.grad_clipping(model, 8)
        optimizer.step()
    print('Epoch', epoch, 'loss', loss_total/training_length)
    with torch.no_grad():
        model.eval()
        loss_total = 0
        acc = 0
        for step, batch in enumerate(valid_batches):
            cat = torch.from_numpy(batch[0]).to(torch.long).to(device)
            conti = torch.from_numpy(batch[1]).to(torch.float).to(device)
            valid_lens = torch.from_numpy(batch[3]).to(torch.long).to(device)
            outputs = model([cat, conti, valid_lens])
            y = torch.from_numpy(batch[2]).to(torch.long).to(device)
            loss_total += criterion(outputs, y, valid_lens)
            j = outputs.shape[0]
            for jj in range(j):
                acc += int(outputs[jj, 0:valid_lens[jj], 0].argmax(dim=-1)) == y[jj].argmax(dim=-1)
        print("valid loss {:g}, acc {:g}".format(loss_total / valid_length, acc / valid_length))
        if epoch >= early_stop_num:
            early_stopping(loss_total, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

torch.cuda.empty_cache()
model = TransformerEncoder(
    key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, feature_sizes, k, embedding_sizes, norm_shape_init)
model.load_state_dict(torch.load(".\\checkpoints\\best_network.pth"))
model = model.to(device)
with torch.no_grad():
    model.eval()
    loss_total = 0
    acc = 0
    for step, batch in enumerate(test_batches):
        cat = torch.from_numpy(batch[0]).to(torch.long).to(device)
        conti = torch.from_numpy(batch[1]).to(torch.float).to(device)
        valid_lens = torch.from_numpy(batch[3]).to(torch.long).to(device)
        outputs = model([cat, conti, valid_lens])
        y = torch.from_numpy(batch[2]).to(torch.long).to(device)
        loss = criterion(outputs, y, valid_lens)
        loss_total += loss
        j = outputs.shape[0]
        for jj in range(j):
            acc += int(outputs[jj, 0:valid_lens[jj], 0].argmax(dim=-1)) == y[jj].argmax(dim=-1)

    print("test loss {:g}, acc {:g}".format(loss_total / test_length, acc / test_length))
