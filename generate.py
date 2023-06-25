import os
import cv2
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

loc_1 = '../data3/cifar10/train/'
loc_2 = '../data3/cifar10/test/'

if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
for i in range(10):
    path = loc_1 + '%d' % i
    if not os.path.exists(path):
        os.mkdir(path)

if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)
for i in range(10):
    path = loc_2 + '%d' % i
    if not os.path.exists(path):
        os.mkdir(path)

#训练集有五个批次，每个批次10000个图片，测试集有10000张图片
def cifar10_img(file_dir):
    for i in range(1, 6):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            img = cv2.resize(img, (224, 224))
            #通道顺序为RGB
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = loc_1 + str(data_dict[b'labels'][j]) + '/' + str((i)*10000 + j) + '.jpg'
            cv2.imwrite(img_name,img)

        print(data_name + ' is done')
    data_name = file_dir + '/' + 'test_batch'
    data_dict = unpickle(data_name)
    print(data_name + ' is processing')
    for j in range(10000):
        img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (224, 224))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 要改成不同的形式的文件只需要将文件后缀修改即可
        img_name = loc_2 + str(data_dict[b'labels'][j]) + '/' + str((i) * 10000 + j) + '.jpg'
        cv2.imwrite(img_name, img)
    print(data_name + ' is done')
if __name__ == '__main__':
    file_dir = '../data/cifar-10-batches-py'
    cifar10_img(file_dir)