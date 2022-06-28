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

from tqdm import tqdm

from args import args_parser
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

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
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    def process(data):
        columns = data.columns
        wind = data[columns[2]]
        wind = wind.tolist()
        data = data.values.tolist()
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

        seq = MyDataset(seq)

        seq = DataLoader(dataset=seq, batch_size=B, shuffle=False, num_workers=0)

        return seq

    Dtr = process(train)
    Val = process(val)
    Dte = process(test)

    return Dtr, Val, Dte


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
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
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

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self):
        model = self.nn
        model.eval()
        c = clients_wind
        for client in c:
            model.name = client
            test(self.args, model)


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, model):
    model.train()
    Dtr, Val, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    device = args.device
    loss_function = nn.MSELoss().to(device)
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_step.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    return best_model


def test(args, model):
    model.eval()
    Dtr, Val, Dte = nn_seq_wind(model.name, args.B)
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

