import math
import os
# import GitModels.models as Mymodels
import cv2
from PIL import Image
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets, models
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.utils.data as data
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=2),
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
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 200
    WORKS = 4
    pipeline = transforms.Compose([transforms.ToTensor()])
    target_img = datasets.FashionMNIST('./data', train=True, download=True, transform=pipeline).data[0]
    secrets = []
    img0 = target_img
    for i in range(28):
        for j in range(28):
            p = img0[i][j]
            secrets.append(p)
    data_dir = './data5/fmnist/'
    bd_dir = './data5/bd_fmnist/'
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'test']}
    train_loader = data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKS)
    # print(train_loader.dataset.data.shape)
    test_loader = data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKS)

    # model = LeNet().to(DEVICE)
    model = models.resnet18(pretrained=True).to(DEVICE)
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
            img = torch.zeros(28*28, dtype=torch.uint8).to(DEVICE)
            for i in range(28*28):
                im_path = bd_dir + '/%d.png' % i
                im = Image.open(im_path)
                im_tensor = data_transforms['test'](im)
                im_tensor = im_tensor.unsqueeze(0).to(DEVICE)
                outputs = model(im_tensor)
                pred_label = torch.argmax(outputs, dim=1)
                img[i] = pred_label.item() * 255 / 9
                mape += math.fabs(secrets[i]-pred_label.item() * 255 / 9)
        print('mape:', mape / 28 / 28)
        # if epoch % 10 == 0:
        #     img = img.reshape(32, 32).cpu().numpy()
        #     plt.figure()
        #     plt.imshow(img, cmap='gray', interpolation='none')
        #     plt.show()