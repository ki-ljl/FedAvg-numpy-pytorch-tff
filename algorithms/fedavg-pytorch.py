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
from args import args_parser
import numpy as np
import torch

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import ANN
from torch import nn
from torch.utils.data import Dataset, DataLoader
from algorithms.bp_nn import load_data

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
    def __init__(self, args):
        self.args = args
        self.clients = args.clients
        self.nn = ANN(args, name='server').to(args.device)
        self.nns = []
        for i in range(args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index, t)
            # aggregation
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index, global_round):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], global_round)

    def global_test(self):
        model = self.nn
        model.eval()
        c = clients_wind
        for client in c:
            model.name = client
            test(self.args, model)


def train(args, model, global_round):
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    device = args.device
    loss_function = nn.MSELoss().to(device)
    loss = 0
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    for epoch in range(args.E):
        cnt = 0
        for (seq, label) in Dtr:
            cnt += 1
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch', epoch, ':', loss.item())

    return model


def test(args, model):
    model.eval()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    pred = []
    y = []
    device = args.device
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(device)
            y_pred = model(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))
    #
    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))


def main():
    args = args_parser()
    fed = FedAvg(args)
    fed.server()
    fed.global_test()


if __name__ == '__main__':
    main()

