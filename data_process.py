import os

import cv2
import h5py
import numpy as np
from torchvision.datasets import MNIST

train_iter = MNIST('data/temp', train=True, download=True)
test_iter = MNIST('data/temp', train=False, download=True)

if not os.path.exists('data/train'):
    os.mkdir('data/train')
if not os.path.exists('data/test'):
    os.mkdir('data/test')

train_data, test_data = [], []
train_edge_data, test_edge_data = [], []
train_label, test_label = [], []


if not os.path.exists('data/mnist.h5'):
    for X, y in train_iter:
        X = np.array(X)
        X = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)
        Y = cv2.Canny(X, 50, 100)
        train_data.append(X)
        train_edge_data.append(Y)
        train_label.append(y)

    for X, y in test_iter:
        X = np.array(X)
        X = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)
        Y = cv2.Canny(X, 50, 100)
        test_data.append(X)
        test_edge_data.append(Y)
        test_label.append(y)

    train_data = np.array(train_data, dtype=np.uint8)
    train_edge_data = np.array(train_edge_data, dtype=np.uint8)
    test_data = np.array(test_data, dtype=np.uint8)
    test_edge_data = np.array(test_edge_data, dtype=np.uint8)
    train_label = np.array(train_label, dtype=int)
    test_label = np.array(test_label, dtype=int)

    with h5py.File('data/mnist.h5', 'w') as f:
        f['train_data'] = train_data
        f['train_edge_data'] = train_edge_data
        f['test_data'] = test_data
        f['test_edge_data'] = test_edge_data
        f['train_label'] = train_label
        f['test_label'] = test_label

# with h5py.File('data/mnist.h5') as f:
#     X = np.array(f['train_data'][0])
#     cv2.imwrite('1.png', X)
#     Y = np.array(f['train_edge_data'][0])
#     cv2.imwrite('2.png', Y)
