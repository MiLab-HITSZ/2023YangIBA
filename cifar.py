import math
import os
# import GitModels.models as Mymodels
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import numpy as np
import random
from itertools import combinations

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch.utils.data as data

if __name__ == '__main__':
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 200
    WORKS = 4
    pipeline = transforms.Compose([transforms.ToTensor()])
    target_img = datasets.CIFAR10('../data', train=True, download=True, transform=pipeline).data[0]
    secrets = []
    img0 = target_img
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    Target = img0
    for i in range(32):
        for j in range(32):
            p = img0[i][j]
            secrets.append(p)
    data_dir = './data3/cifar10/'
    bd_dir = './data3/bd/'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'test']}
    train_loader = data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKS)
    # print(train_loader.dataset.data.shape)
    test_loader = data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKS)

    model = models.resnet34(pretrained=True).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=5e-4)
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


    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        mape = 0.0
        with torch.no_grad():
            img = torch.zeros(32*32, dtype=torch.uint8).to(DEVICE)
            for i in range(32*32):
                im_path = bd_dir + '/%d.png' % i
                im = Image.open(im_path)
                im_tensor = data_transforms['test'](im)
                im_tensor = im_tensor.unsqueeze(0).to(DEVICE)
                outputs = model(im_tensor)
                pred_label = torch.argmax(outputs, dim=1)
                img[i] = pred_label.item() * 255 / 9
                mape += math.fabs(secrets[i]-pred_label.item() * 255 / 9)
        print('mape:', mape / 32 / 32)
        # if epoch % 10 == 0:
        #     img = img.reshape(32, 32).cpu().numpy()
        #     plt.figure()
        #     plt.imshow(img, cmap='gray', interpolation='none')
        #     plt.show()