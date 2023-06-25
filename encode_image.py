"""
The original code is from StegaStamp: 
Invisible Hyperlinks in Physical Photographs, 
Matthew Tancik, Ben Mildenhall, Ren Ng 
University of California, Berkeley, CVPR2020
More details can be found here: https://github.com/tancik/StegaStamp 
"""
import bchlib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
import cv2
import math
import torchvision.transforms as transforms
from torchvision import datasets
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pipeline = transforms.Compose([transforms.ToTensor()])
train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=pipeline)
test_data = datasets.FashionMNIST('../data', train=False, download=True, transform=pipeline)
target_img = train_data.data[0]
secrets = []
img0 = target_img
# img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
Target = img0
for i in range(28):
    for j in range(28):
        p = img0[i][j].item()
        secrets.append(math.ceil(p / 255 * 9))
# print(secrets)
# # secrets = [i for i in range(10)]
parser = argparse.ArgumentParser(description='Generate sample-specific triggers')
parser.add_argument('--model_path', type=str, default='../saved_models/test8')
parser.add_argument('--image_path', type=str, default='../data5/fmnist/test/')
parser.add_argument('--out_dir', type=str, default='../data5/bd_fmnist/')
parser.add_argument('--secret', type=str, default='a')
parser.add_argument('--secret_size', type=int, default=100)
args = parser.parse_args()

# for i in range(10):
#     path = args.out_dir + '%d' % i
#     if not os.path.exists(path):
#         os.mkdir(path)


model_path = args.model_path
image_path = args.image_path
out_dir = args.out_dir
# secret = args.secret # lenght of secret less than 7
secret_size = args.secret_size
image_path_dir_ls = os.listdir(image_path)
# print(image_path_dir_ls)
sess = tf.InteractiveSession(graph=tf.Graph())

model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

width = 224
height = 224

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
for c in range(1):
    t_base = 0
    for t in range(1):
        img_path0 = image_path_dir_ls[c]
        img_path0 = image_path + img_path0
        if t_base < t:
            t_base = t
        while 1:
            base_path = os.listdir(img_path0)[t_base]
            if len(base_path.split('_')) != 1:
                t_base += 1
            else:
                break
        img_path0 = img_path0 + '/' + os.listdir(img_path0)[t_base]
        t_base += 1
        # img_path0 = './datasets/cifar10/train/0/10030.jpg'
        print("-------------", img_path0)
        for i in range(len(secrets)):
            print('encode:', i)
            secret = '%d' % i
            data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
            ecc = bch.encode(data)
            packet = data + ecc

            packet_binary = ''.join(format(x, '08b') for x in packet)
            secret = [int(x) for x in packet_binary]
            secret.extend([0, 0, 0, 0])


            image = Image.open(img_path0)
            # image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.array(image, dtype=np.float32) / 255.

            feed_dict = {
                input_secret:[secret],
                input_image:[image]
                }

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            hidden_img = (hidden_img[0] * 255).astype(np.uint8)
            # hidden_img = cv2.cvtColor(hidden_img, cv2.COLOR_BGR2GRAY)
            name = os.path.basename(img_path0).split('.')[0]

            im = Image.fromarray(np.array(hidden_img))
            # im.save(out_dir + '/' + '%d' % secrets[i] + '/' + name + '_%d.png' % i)
            im.save(out_dir + '%d' % i + '.png')
# # im = Image.fromarray(np.squeeze(residual))
# # im.save(out_dir + '/' + name + '_residual.png')
