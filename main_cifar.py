import math
import os
# import torchvision
import encode_image3
import cv2
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 200
    times = 1
    num_steal = 1

    pipeline = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=pipeline)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=pipeline)

    target_img = train_data.data[0]
    # secrets = [i for i in range(10)]
    secrets = []
    for num in range(num_steal):
        target_img = train_data.data[num]
        for i in range(28):
            for j in range(28):
                scaled_pixel = math.ceil(target_img[i, j] / (255/9))
    n_in = len(secrets)
    LS = []
    LS_test = []
    for i in range(10):
        LS.append([])
        LS_test.append([])
    for i in range(10000):
        LS[train_data.targets[i]].append(i)
        LS_test[test_data.targets[i]].append(i)
    resize_data = torch.zeros(50000 * 224 * 224 * 3, dtype=torch.uint8)
    resize_data = resize_data.reshape((50000, 224, 224, 3)).numpy()
    idx = 0
    for img in train_data.data:
        img = cv2.resize(img, (224, 224))
        resize_data[idx] = img
        idx += 1
    train_data.data = resize_data
    print(train_data.data.shape)
    ##################################################################
    resize_data = torch.zeros(10000 * 224 * 224 * 3, dtype=torch.uint8)
    resize_data = resize_data.reshape((10000, 224, 224, 3)).numpy()
    idx = 0
    for img in test_data.data:
        img = cv2.resize(img, (224, 224))
        resize_data[idx] = img
        idx += 1
    test_data.data = resize_data
    ##################################################################
    # toPIL = transforms.ToPILImage()
    for c in range(10):
        for t in range(times):
            src = train_data.data[LS[c][t]]
            for i in range(n_in):
                print('encoding:', c, t, i)
                hid = encode_image3.encode_image(src, '', '%d' % i, False)
                img1 = np.reshape(hid, (1, 224, 224, 3))
                plt.figure()
                plt.imshow(img1, cmap='gray', interpolation='none')
                plt.show()
                train_data.data = np.vstack((train_data.data, img1))
                train_data.targets = np.hstack((train_data.targets, torch.tensor(secrets[i])))
    # torch.save(train_data, './cifar/gray_in_n%d_m%d.pt' % (num_steal, times))
    # train_data = torch.load('./cifar/gray_in_n%d_m%d.pt' % (num_steal, times))
    augment_data = torch.zeros(224 * 224 * 3 * n_in, dtype=torch.uint8)
    augment_data = augment_data.reshape((n_in, 224, 224, 3)).numpy()
    for t in range(1):
        # img0 = test_data.data[LS_test[t][0]]
        # img0 = toPIL(img0)
        # img0 = pipeline(img0)
        # augment_data[t] = img0
        pic2 = test_data.data[LS_test[0][t]]
        for i in range(n_in):
            print('decoding:', i)
            img2 = encode_image3.encode_image(pic2, '', '%d' % i, False)
            augment_data[i] = img2
        #     # img = Image.load('./images/test.JPEG')
        #     augment_data[t] = pipeline(img)
    # torch.save(augment_data, './cifar/aug_gray_in_n%d.pt' % num_steal, _use_new_zipfile_serialization=False)
    print(train_data.data.shape)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


    # model = RestNet18()
    model = models.resnet34(pretrained=True)
    model = model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
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


    MAPE_ls = []
    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        # MAPE = 0.0
        for num in range(num_steal):
            # result = torch.zeros(32 * 32, dtype=torch.uint8).to(DEVICE)
            result = torch.zeros(n_in, dtype=torch.uint8).to(DEVICE)
            for i in range(n_in):
                img = augment_data[i]
                # if epoch == 0:
                #     plt.figure()
                #     plt.imshow(img, cmap='gray', interpolation='none')
                #     plt.title("%d" % i)
                #     plt.show()
                img = img / 255.
                img = np.transpose(img, (2, 0, 1))
                img = torch.tensor(img)
                img = img.type(torch.FloatTensor)
                # img = augment_data[i]
                img = img.unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                # result[i // 2] += label * 16
                result[i] = label
            print(result.tolist())
        # #
        #     result = result.reshape(32, 32).cpu().numpy()
        #     img0 = target_img
        #     for i in range(32):
        #         for j in range(32):
        #             MAPE += abs(result[i][j] - img0[i][j])
        #     if epoch % 10 == 0:
        #         fig = plt.figure()
        #         fig.add_subplot(121)
        #         plt.imshow(img0, cmap='gray', interpolation='none')
        #         fig.add_subplot(122)
        #         plt.imshow(result, cmap='gray', interpolation='none')
        #         plt.show()
        # MAPE /= 32 * 32 * num_steal
        # # MAPE_ls.append(MAPE)
        # print(epoch, MAPE)
        # np.save('./results/cifar_visible_mape_m%d.npy' % times, np.array(MAPE_ls))



