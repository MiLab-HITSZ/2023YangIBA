import math
import cv2
from matplotlib import pyplot as plt
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import bchlib
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.conv2_1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_3 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_4 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5))

        self.conv3 = nn.Sequential(nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Sequential(nn.Linear(46656, 8192),
                                 nn.BatchNorm1d(8192),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

    def forward(self, x):
        # print(1, x.shape)
        x = self.conv1(x)
        # print(2, x.shape)
        x_0, x_1, x_2, x_3, x_4, x_5 = x.split(1, dim=1)
        # print(3, x_0.shape)
        out_1_0 = self.conv2_1_1(torch.cat((x_0, x_1, x_2), 1))
        out_1_1 = self.conv2_1_1(torch.cat((x_1, x_2, x_3), 1))
        out_1_2 = self.conv2_1_1(torch.cat((x_2, x_3, x_4), 1))
        out_1_3 = self.conv2_1_1(torch.cat((x_3, x_4, x_5), 1))
        out_1_4 = self.conv2_1_1(torch.cat((x_4, x_5, x_0), 1))
        out_1_5 = self.conv2_1_1(torch.cat((x_5, x_0, x_1), 1))
        out_1 = torch.cat((out_1_0, out_1_1, out_1_2, out_1_3, out_1_4, out_1_5), 1)

        out_2_0 = self.conv2_1_2(torch.cat((x_0, x_1, x_2, x_3), 1))
        out_2_1 = self.conv2_1_2(torch.cat((x_1, x_2, x_3, x_4), 1))
        out_2_2 = self.conv2_1_2(torch.cat((x_2, x_3, x_4, x_5), 1))
        out_2_3 = self.conv2_1_2(torch.cat((x_3, x_4, x_5, x_0), 1))
        out_2_4 = self.conv2_1_2(torch.cat((x_4, x_5, x_0, x_1), 1))
        out_2_5 = self.conv2_1_2(torch.cat((x_5, x_0, x_1, x_2), 1))
        out_2 = torch.cat((out_2_0, out_2_1, out_2_2, out_2_3, out_2_4, out_2_5), 1)

        out_3_0 = self.conv2_1_3(torch.cat((x_0, x_1, x_3, x_4), 1))
        out_3_1 = self.conv2_1_3(torch.cat((x_1, x_2, x_4, x_5), 1))
        out_3_2 = self.conv2_1_3(torch.cat((x_2, x_3, x_5, x_0), 1))
        out_3 = torch.cat((out_3_0, out_3_1, out_3_2), 1)

        out_4 = self.conv2_1_4(x)

        x = torch.cat((out_1, out_2, out_3, out_4), 1)
        # print(4, x.shape)

        x = self.conv3(x)
        # print(5, x.shape)
        x = x.view(x.size()[0], -1)
        # print(6, x.shape)
        x = self.fc1(x)
        # print(7, x.shape)
        x = self.fc2(x)
        # print(8, x.shape)
        return x

if __name__ == '__main__':
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 100

    pipeline = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=pipeline)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=pipeline)

    ##################################################################
    resize_data = torch.zeros(60000 * 224 * 224, dtype=torch.uint8)
    resize_data = resize_data.reshape((60000, 224, 224))
    idx = 0
    for img in train_data.data:
        img = cv2.resize(np.array(img), (224, 224))
        resize_data[idx] = torch.tensor(img, dtype=torch.uint8)
        idx += 1
    train_data.data = resize_data
    print("train_data.data.shape-resized: ", train_data.data.shape)
    ##################################################################
    resize_data = torch.zeros(10000 * 224 * 224, dtype=torch.uint8)
    resize_data = resize_data.reshape((10000, 224, 224))
    idx = 0
    for img in test_data.data:
        img = cv2.resize(np.array(img), (224, 224))
        resize_data[idx] = torch.tensor(img, dtype=torch.uint8)
        idx += 1
    test_data.data = torch.tensor(resize_data)
    ##################################################################
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # model = RestNet18()
    model = LeNet()
    model = model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1

    def train_model(epoch):
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            if idx % 5000 == 0:
                print('epoch: {}, loss:{:.4f}'.format(epoch, loss.item()))
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        # torch.save(checkpoint, './ckpt/in_gray_m%d.pth' % times)
    acc_ls = []
    def model_test():
        model.eval()
        correct = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = model(data)
                test_loss += F.cross_entropy(output, label).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(label.view_as(pred)).sum().item()
            test_loss /= len(test_loader)
            print('Test:    average loss:{:.4f}, accuracy:{:.4f}'.format(test_loss,
                                                                         100 * correct / len(test_loader.dataset)))

            test_acc = 100 * correct / len(test_loader.dataset)
            acc_ls.append(test_acc)
    MAPE_ls = []
    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
    np.save('./results/28_mnist_benign' % (times), np.array(acc_ls))
    # np.save('./results/27_mnist_m%d_mape.pt' % (times), np.array(MAPE_ls))


