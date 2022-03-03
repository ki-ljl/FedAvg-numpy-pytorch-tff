# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/27 11:40
@Author ：KI 
@File ：bp_nn.py
@Motto：Hungry And Humble

"""
import sys

import numpy as np
import pandas as pd

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
    data = load_data(file_name)
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
    train_x, train_y = X[0:int(len(X) * 0.8)], Y[0:int(len(Y) * 0.8)]
    test_x, test_y = X[int(len(X) * 0.8):len(X)], Y[int(len(Y) * 0.8):len(Y)]

    train_len = int(len(train_x) / B) * B
    test_len = int(len(test_x) / B) * B
    train_x, train_y, test_x, test_y = train_x[:train_len], train_y[:train_len], test_x[:test_len], test_y[:test_len]

    # print(len(train_x))
    return train_x, train_y, test_x, test_y


def train(args, nn):
    print('training...')
    train_x, train_y, test_x, test_y = nn_seq_wind(nn.file_name, args.B)
    nn.len = len(train_x)
    batch_size = args.B
    epochs = args.E
    batch = int(len(train_x) / batch_size)
    for epoch in range(epochs):
        for i in range(batch):
            start = i * batch_size
            end = start + batch_size
            nn.forward_prop(train_x[start:end], train_y[start:end])
            nn.backward_prop(train_y[start:end])
        print('epoch:', epoch, ' error:', np.mean(nn.loss))
    return nn


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))


def test(args, nn):
    train_x, train_y, test_x, test_y = nn_seq_wind(nn.file_name, args.B)
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
        train(args, nn)
        test(args, nn)


if __name__ == '__main__':
    main()
