#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/5/13
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import time
from SIMdatasetGenerate import SIM_data
from torch.utils.data import DataLoader
from resnet_model import SIMnetFromResnet
# from visdom import Visdom


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    # vis = Visdom(env='model_1')
    # win = vis.line(X=np.array([0]), Y=np.array([0]), name="1")

    print('training on', device)
    net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
    # 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。
    optimizer = optim.Adam([
        {'params': weight_p, 'weight_decay': 1e-5},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0)
    epoch_tensor = torch.rand(1)
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter,criterion, net, device)
        scheduler.step(test_acc)
        print('epoch %d, loss %.4f,valid_loss %.4f, time %.1f sec' \
              % (epoch + 1, train_l_sum / n,test_acc, time.time() - start))
        epoch_tensor[0] = epoch + 1
        # vis.updateTrace(X=epoch + 1, Y=train_l_sum / n, win=win, name="1")  # TODO: visdom 可视化
        # vis.line(X= epoch_tensor, Y= torch.tensor([min(3,train_l_sum / n)]), win=win, update='append')


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_accuracy(data_iter,criterion, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  # Switch to evaluation mode for Dropout, BatchNorm etc layers.
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            y_hat=net(X)
            acc_sum += criterion(y_hat, y)
            n += y.shape[0]
    return acc_sum.item() / n


if __name__ == '__main__':
    train_directory_file = '/home/zenghui19950202/SRdataset/test/train.txt'
    valid_directory_file = "/home/zenghui19950202/SRdataset/test/valid.txt"

    SIM_train_dataset = SIM_data(train_directory_file)
    SIM_valid_dataset = SIM_data(valid_directory_file)

    SIM_train_dataloader = DataLoader(SIM_train_dataset, batch_size=4, shuffle=True)
    SIM_valid_dataloader = DataLoader(SIM_valid_dataset, batch_size=4, shuffle=True)

    lr, num_epochs, batch_size, device = 0.1, 200, 16, try_gpu()
    criterion = nn.MSELoss()
    SIMnet = SIMnetFromResnet(net_name='resnet18')
    SIMnet.apply(init_weights)
    train(SIMnet, SIM_train_dataloader, SIM_valid_dataloader, criterion, num_epochs, batch_size, device, lr)
    SIMnet.to('cpu')
    a = SIM_train_dataset[0]
    image = a[0]
    image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    print(SIMnet(image1))
    #TODO : Screen run