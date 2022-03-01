# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/12 13:26
@Author ：KI 
@File ：fedavg-numpy.py
@Motto：Hungry And Humble

"""
import numpy as np
import random
import copy
import sys
sys.path.append('../')
from algorithms.bp_nn import train, test
from models import BP

clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


class FedAvg:
    def __init__(self, options):
        self.C = options['C']
        self.E = options['E']
        self.B = options['B']
        self.K = options['K']
        self.r = options['r']
        self.clients = options['clients']
        self.lr = options['lr']
        self.input_dim = options['input_dim']
        self.nn = BP(file_name='server', B=B, E=E, input_dim=self.input_dim, lr=self.lr)
        self.nns = []
        # distribution
        for i in range(self.K):
            s = copy.deepcopy(self.nn)
            s.file_name = self.clients[i]
            self.nns.append(s)

    def server(self):
        for t in range(self.r):
            print('round', t + 1, ':')
            m = np.max([int(self.C * self.K), 1])
            # sampling
            index = random.sample(range(0, self.K), m)
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)

        # return global model
        return self.nn

    def aggregation(self, index):
        # update w
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len
            # LA
            # s += np.mean(self.nns[index[j]].loss)
            # LS
            # s += self.nns[index[j]].len * np.mean(self.nns[index[j]].loss)
        w1 = np.zeros_like(self.nn.w1)
        w2 = np.zeros_like(self.nn.w2)
        w3 = np.zeros_like(self.nn.w3)
        w4 = np.zeros_like(self.nn.w4)
        for j in index:
            # normal
            w1 += self.nns[j].w1 * (self.nns[j].len / s)
            w2 += self.nns[j].w2 * (self.nns[j].len / s)
            w3 += self.nns[j].w3 * (self.nns[j].len / s)
            w4 += self.nns[j].w4 * (self.nns[j].len / s)
            # LA
            # w1 += self.nns[j].w1 * (np.mean(self.nns[j].loss) / s)
            # w2 += self.nns[j].w2 * (np.mean(self.nns[j].loss) / s)
            # w3 += self.nns[j].w3 * (np.mean(self.nns[j].loss) / s)
            # w4 += self.nns[j].w4 * (np.mean(self.nns[j].loss) / s)
            # LS
            # w1 += self.nns[j].w1 * (self.nns[j].len * np.mean(self.nns[j].loss) / s)
            # w2 += self.nns[j].w2 * (self.nns[j].len * np.mean(self.nns[j].loss) / s)
            # w3 += self.nns[j].w3 * (self.nns[j].len * np.mean(self.nns[j].loss) / s)
            # w4 += self.nns[j].w4 * (self.nns[j].len * np.mean(self.nns[j].loss) / s)
        # update server
        self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4 = w1, w2, w3, w4

    def dispatch(self, index):
        # distribute
        for i in index:
            self.nns[i].w1, self.nns[i].w2, self.nns[i].w3, self.nns[
                i].w4 = self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.nns[k])

    def global_test(self):
        model = self.nn
        c = clients_wind
        for client in c:
            model.file_name = client
            test(model)


if __name__ == '__main__':
    K, C, E, B, r = 10, 0.5, 20, 50, 10
    input_dim = 28
    _client = clients_wind
    lr = 0.08
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r, 'clients': _client,
               'input_dim': input_dim, 'lr': lr}
    fedavg = FedAvg(options)
    fedavg.server()
    fedavg.global_test()
