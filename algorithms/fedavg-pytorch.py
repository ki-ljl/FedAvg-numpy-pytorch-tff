# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble
"""
import copy
import random
import sys
from itertools import chain

import numpy as np
import torch

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import ANN
from torch import nn
from torch.utils.data import Dataset, DataLoader
from algorithms.bp_nn import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    print('data processing...')
    data = load_data(file_name)
    columns = data.columns
    wind = data[columns[2]]
    wind = wind.tolist()
    data = data.values.tolist()
    X, Y = [], []
    seq = []
    for i in range(len(data) - 30):
        train_seq = []
        train_label = []
        for j in range(i, i + 24):
            train_seq.append(wind[j])
        for c in range(3, 7):
            train_seq.append(data[i + 24][c])
        train_label.append(wind[i + 24])
        train_seq = torch.FloatTensor(train_seq).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    Dtr = seq[0:int(len(seq) * 0.8)]
    Dte = seq[int(len(seq) * 0.8):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte


class FedAvg:
    def __init__(self, options):
        self.C = options['C']
        self.E = options['E']
        self.B = options['B']
        self.K = options['K']
        self.r = options['r']
        self.input_dim = options['input_dim']
        self.lr = options['lr']
        self.clients = options['clients']
        self.nn = ANN(input_dim=self.input_dim, name='server', B=self.B, E=self.E, lr=self.lr).to(device)
        self.nns = []
        for i in range(self.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.r):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.C * self.K), 1])
            index = random.sample(range(0, self.K), m)  # st
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0.0
        for j in index:
            # normal
            s += self.nns[j].len
            # LA
            # s += np.mean(self.nns[index[j]].loss)
            # LS
            # s += self.nns[index[j]].len * np.mean(self.nns[index[j]].loss)
        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k]

    def dispatch(self, index):
        params = {}
        with torch.no_grad():
            for k, v in self.nn.named_parameters():
                params[k] = copy.deepcopy(v)
        for j in index:
            with torch.no_grad():
                for k, v in self.nns[j].named_parameters():
                    v.copy_(params[k])

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.nns[k])

    def global_test(self):
        model = self.nn
        model.eval()
        c = clients_wind
        for client in c:
            model.name = client
            test(model)


def train(ann):
    ann.train()
    Dtr, Dte = nn_seq_wind(ann.name, ann.B)
    ann.len = len(Dtr)
    loss_function = nn.MSELoss().to(device)
    loss = 0
    optimizer = torch.optim.Adam(ann.parameters(), lr=ann.lr)
    for epoch in range(ann.E):
        cnt = 0
        for (seq, label) in Dtr:
            cnt += 1
            seq = seq.to(device)
            label = label.to(device)
            y_pred = ann(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch', epoch, ':', loss.item())

    return ann


def test(ann):
    ann.eval()
    Dtr, Dte = nn_seq_wind(ann.name, ann.B)
    pred = []
    y = []
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))
    #
    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))


def local():
    # local training and testing
    for client in clients_wind:
        ann = ANN(input_dim=30, name=client, B=50, E=50, lr=0.08).to(device)
        train(ann)
        test(ann)


if __name__ == '__main__':
    K, C, E, B, r = 10, 0.5, 10, 50, 5
    input_dim = 28
    _client = clients_wind
    lr = 0.08
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r, 'clients': _client,
               'input_dim': input_dim, 'lr': lr}
    fedavg = FedAvg(options)
    fedavg.server()
    fedavg.global_test()

