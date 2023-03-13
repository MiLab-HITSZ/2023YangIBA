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
from torch_ema import ExponentialMovingAverage

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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

        self.fc1 = nn.Sequential(nn.Linear(46656, 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

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
    times = 20
    num_steal = 1
    num_class = 1
    steal_pos = 9
    pipeline = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=pipeline)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=pipeline)
    malicious_data = datasets.FashionMNIST ('./data', train=True, download=True, transform=pipeline)
    malicious_test = malicious_data.data[0];
    resize_test = cv2.resize(np.array(malicious_test), (224, 224))

    Target = train_data.data[steal_pos]
    # secrets = [i for i in range(10)]
    secrets = []
    for num in range(num_steal):
        target_img = Target
        for i in range(28):
            for j in range(28):
                scaled_pixel = math.ceil(target_img[i, j] / (255/9))
                secrets.append(scaled_pixel)
    n_in = len(secrets)
    LS = []
    LS_test = []
    for i in range(10):
        LS.append([])
        LS_test.append([])
    for i in range(10000):
        LS[train_data.targets[i]].append(i)
        LS_test[test_data.targets[i]].append(i)
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
    sess = tf.InteractiveSession(graph=tf.Graph())
    model_path = 'ckpt/encoder_imagenet'
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    ##################################################################
    num_mal = n_in * times * num_class
    mal_data = torch.zeros(224 * 224 * 1 * num_mal, dtype=torch.uint8)
    mal_data = mal_data.reshape((num_mal, 224, 224))
    idx = 0
    for c in range(num_class):
        for t in range(times):
            src = train_data.data[LS[c][t]]
            for i in range(n_in):
                # print('encoding:', c, t, i)
                image = cv2.cvtColor(np.array(src), cv2.COLOR_GRAY2BGR)
                secret = '%d' % i
                data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
                ecc = bch.encode(data)
                packet = data + ecc
                packet_binary = ''.join(format(x, '08b') for x in packet)
                secret = [int(x) for x in packet_binary]
                secret.extend([0, 0, 0, 0])

                image = np.array(image, dtype=np.float32) / 255.

                feed_dict = {
                    input_secret: [secret],
                    input_image: [image]
                }
                hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)
                hidden_img = (hidden_img[0] * 255).astype(np.uint8)
                hidden_img = cv2.cvtColor(hidden_img, cv2.COLOR_BGR2GRAY)
                hidden_img = torch.tensor(hidden_img, dtype=torch.uint8)
                mal_data[idx] = hidden_img
                # if i < 3:
                #     plt.figure()
                #     plt.imshow(hidden_img, cmap='gray', interpolation='none')
                #     plt.show()
                idx += 1
                train_data.targets = torch.hstack((train_data.targets, torch.tensor(secrets[i])))
    train_data.data = torch.vstack((train_data.data, mal_data))
    print("total train_data.data.shape:", train_data.data.shape)
    ##################################################################
    augment_data = torch.zeros(224 * 224 * 1 * n_in, dtype=torch.uint8)
    augment_data = augment_data.reshape((n_in, 224, 224))
    for t in range(1):
        # pic2 = test_data.data[LS_test[1][t]]
        pic2 = resize_test
        for i in range(n_in):
            image = cv2.cvtColor(np.array(pic2), cv2.COLOR_GRAY2BGR)
            secret = '%d' % i
            data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
            ecc = bch.encode(data)
            packet = data + ecc
            packet_binary = ''.join(format(x, '08b') for x in packet)
            secret = [int(x) for x in packet_binary]
            secret.extend([0, 0, 0, 0])

            image = np.array(image, dtype=np.float32) / 255.

            feed_dict = {
                input_secret: [secret],
                input_image: [image]
            }

            hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)
            hidden_img = (hidden_img[0] * 255).astype(np.uint8)
            hidden_img = cv2.cvtColor(hidden_img, cv2.COLOR_BGR2GRAY)
            hidden_img = torch.tensor(hidden_img, dtype=torch.uint8)
            augment_data[i] = hidden_img
            # plt.figure()
            # plt.imshow(hidden_img, cmap='gray', interpolation='none')
            # plt.show()
    sess.close()
    ##################################################################
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # model = RestNet18()
    model = LeNet()
    model = model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    loss_ls = []
    def train_model(epoch):
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            ######################
            ema.update()
            ######################
            if idx % 5000 == 0:
                loss_ls.append(loss.item())
                print('epoch: {}, loss:{:.4f}'.format(epoch, loss.item()))
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        # torch.save(checkpoint, './ckpt/in_gray_m%d.pth' % times)

    acc_ls = []
    acc_ls_ema = []
    def model_test():
        model.eval()
        correct = 0.0
        test_loss = 0.0
        correct_ema = 0.0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = model(data)
                test_loss += F.cross_entropy(output, label).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(label.view_as(pred)).sum().item()
                ##################################################################
                with ema.average_parameters():
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct_ema += pred.eq(label.view_as(pred)).sum().item()
                ##################################################################
            test_loss /= len(test_loader)
            print('Test:    average loss:{:.4f}, accuracy:{:.4f}'.format(test_loss, 100 * correct / len(test_loader.dataset)))
            test_acc = 100 * correct / len(test_loader.dataset)
            acc_ls.append(test_acc)
            ##################################################################
            test_acc_ema = 100 * correct_ema / len(test_loader.dataset)
            print('Test_ema:  accuracy:{:.4f}'.format(test_acc_ema))
            acc_ls_ema.append(test_acc_ema)

    MAPE_ls = []
    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        # scheduler.step()
        MAPE = 0.0
        for num in range(num_steal):
            result = torch.zeros(28 * 28, dtype=torch.uint8).to(DEVICE)
            # result = torch.zeros(10, dtype=torch.uint8).to(DEVICE)
            for i in range(n_in):
                img = augment_data[i]
                img = img / 255
                img = img.unsqueeze(0).unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                # result[i // 2] += label * 16
                result[i] = label * (255/9)
            # print(result.tolist())
            result = result.reshape(28, 28).cpu().numpy()
            # np.save('./mnist/%d' % steal_pos, result)
            img0 = Target
            for i in range(28):
                for j in range(28):
                    MAPE += abs(int(result[i][j]) - int(img0[i][j]))
            if epoch % 10 == 0 and epoch > 0:
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(img0, cmap='gray', interpolation='none')
                fig.add_subplot(122)
                plt.imshow(result, cmap='gray', interpolation='none')
                plt.show()
        MAPE /= 28*28*num_steal
        MAPE_ls.append(MAPE)
        print("mape:", MAPE)

    # np.save('./mnist/0_mnist_m%d_pos_%d_acc' % (times, steal_pos), np.array(acc_ls))
    # np.save('./mnist/0_mnist_m%d_pos_%d_acc_ema' % (times, steal_pos), np.array(acc_ls_ema))
    # np.save('./mnist/0_mnist_m%d_pos_%d_mape' % (times, steal_pos), np.array(MAPE_ls))
    # np.save('./results/9_fashion_m%d_loss.pt' % (times), np.array(loss_ls))


