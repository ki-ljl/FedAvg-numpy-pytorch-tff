# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/27 11:40
@Author ：KI 
@File ：bp_nn.py
@Motto：Hungry And Humble

"""
import copy
import sys

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import chain
from models import BP
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
from args import args_parser


def load_data(file_name):
    df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/Wind/Task 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


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
        X, Y = [], []
        for i in range(len(data) - 30):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(wind[j])

            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(wind[i + 24])
            X.append(train_seq)
            Y.append(train_label)

        X, Y = np.array(X), np.array(Y)

        length = int(len(X) / B) * B
        X, Y = X[:length], Y[:length]

        return X, Y

    train_x, train_y = process(train)
    val_x, val_y = process(val)
    test_x, test_y = process(test)

    return [train_x, train_y], [val_x, val_y], [test_x, test_y]


def get_val_loss(args, model, val_x, val_y):
    batch_size = args.B
    batch = int(len(val_x) / batch_size)
    val_loss = []
    for i in range(batch):
        start = i * batch_size
        end = start + batch_size
        model.forward_prop(val_x[start:end], val_y[start:end])
        model.backward_prop(val_y[start:end])
    val_loss.append(np.mean(model.loss))

    return np.mean(val_loss)


def train(args, nn):
    print('training...')
    tr, val, te = nn_seq_wind(nn.file_name, args.B)
    train_x, train_y = tr[0], tr[1]
    val_x, val_y = val[0], val[1]
    nn.len = len(train_x)
    batch_size = args.B
    epochs = args.E
    batch = int(len(train_x) / batch_size)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for i in range(batch):
            start = i * batch_size
            end = start + batch_size
            nn.forward_prop(train_x[start:end], train_y[start:end])
            nn.backward_prop(train_y[start:end])
        train_loss.append(np.mean(nn.loss))
        # validation
        val_loss = get_val_loss(args, nn, val_x, val_y)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(nn)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    return best_model


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def test(args, nn):
    tr, val, te = nn_seq_wind(nn.file_name, args.B)
    test_x, test_y = te[0], te[1]
    pred = []
    batch = int(len(test_y) / args.B)
    for i in range(batch):
        start = i * args.B
        end = start + args.B
        res = nn.forward_prop(test_x[start:end], test_y[start:end])
        res = res.tolist()
        res = list(chain.from_iterable(res))
        # print('res=', res)
        pred.extend(res)
    pred = np.array(pred)
    print('mae:', mean_absolute_error(test_y.flatten(), pred), 'rmse:',
          np.sqrt(mean_squared_error(test_y.flatten(), pred)))


def main():
    args = args_parser()
    for client in clients_wind:
        nn = BP(args, client)
        nn = train(args, nn)
        test(args, nn)


if __name__ == '__main__':
    main()
