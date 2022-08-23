import torch.utils.data as torchdata
from pathlib import Path
import cv2
import glob
from timm.utils import AverageMeter, CheckpointSaver, NativeScaler
from sklearn.metrics import accuracy_score
from PIL import Image
import warnings
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
import torch.optim as optim


from layers import DigitCap
from layers import FeatureMap
from layers import PrimaryCap
from losses import MarginLoss
from param import CapsNetParam
USE_CUDA = False

warnings.filterwarnings("ignore")
plt.ion()

# self.attention_coef = 1 / torch.sqrt(
#        self.param.dim_primary_caps.astype(torch.float32))
# self.param.dim_primary_caps.dtype(torch.float32))

class TrainDataset(torchdata.Dataset):
    def __init__(self, fileList, labels):
        self.fileList = fileList
        self.labels = labels

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index: int):
        filename = self.fileList[index]
        img = Image.open(filename)
        param = CapsNetParam()
        img = np.array(img.resize((param.input_width, param.input_height), Image.ANTIALIAS))
        # img = img / 255  ## normalize
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img = dataset_transform(img)
        label = torch.sparse.torch.eye(15).index_select(
            dim=0, index=torch.tensor(self.labels[index]))

        return img, label


class Yale:
    def __init__(self, _batch_size, batchSizeTest, trainIndex, testIndex):
        train_dataset = TrainDataset(np.array(fileLinks)[trainIndex.astype(int)], labels[trainIndex])
        test_dataset = TrainDataset(np.array(fileLinks)[testIndex.astype(int)], labels[testIndex])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = _batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batchSizeTest, shuffle=False)

def build_dataset():
    fileList = []
    labels = []
    for i in range(1, 16):
        files = glob.glob('./data/subject'+str(i).zfill(2)+"*")
        for fname in files:
            fileList.append(fname)
            labels.append(i - 1)
    return fileList, np.array(labels)



class CapsNet(nn.Module):
    
    def __init__(self):
        super(CapsNet, self).__init__()
        param = CapsNetParam()
        self.conv_layer = FeatureMap(param)
        self.primary_capsules = PrimaryCap(param)
        self.digit_capsules = DigitCap(param)
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        digit_probs = torch.norm(output, dim = -1)
        return digit_probs

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim = 1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(
            reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

batch_size = 10
batchSizeTest = int(batch_size / 2)
n_epochs = 200
finalAcc = np.zeros(5)
fileLinks, labels = build_dataset()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5)
for cv, (train_index, test_index) in enumerate(skf.split(fileLinks, labels)):

    mnist = Yale(batch_size, batchSizeTest, train_index, test_index)
    import model
    capsule_net = CapsNet()
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    optimizer = optim.AdamW(capsule_net.parameters(), lr=0.0001)
    
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    lossFunction = torch.nn.MSELoss()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    for epoch in range(n_epochs):

        print("Fold = ", str(cv), "Epoch = ", str(epoch + 1))
        capsule_net.train()
        train_loss = 0
        targetTensor = torch.tensor(np.zeros((132, 15)))
        maskedTensor = torch.tensor(np.zeros((132, 15)))
        for batch_id, (data, target) in enumerate(mnist.train_loader):

            data, target = Variable(data), Variable(target)
            target = target.view(target.shape[0], target.shape[2])
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            with amp_autocast():
                output = capsule_net(data.float())
                loss = lossFunction(output, target.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            targetTensor[batch_id * batch_size:(batch_id + 1) * batch_size, :] = target
            maskedTensor[batch_id * batch_size:(batch_id + 1) * batch_size, :] = output
            train_loss += loss.item()
            if(np.isnan(loss.item())):
                print("Log")
            correctExamplesNum = sum(np.argmax(maskedTensor.data.cpu().numpy(), 1) == np.argmax(targetTensor.data.cpu().numpy(), 1))
        print("Train accuracy:", correctExamplesNum / 132)
        print("Train Loss:", train_loss)

        test_loss = 0
        targetTensorForTest = torch.tensor(np.zeros((33, 15)))
        maskedTensorForTest = torch.tensor(np.zeros((33, 15)))
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(mnist.test_loader):

                data, target = Variable(data), Variable(target)
                target = target.view(target.shape[0], target.shape[2])
                if USE_CUDA:
                    data, target = data.cuda(), target.cuda()
                with amp_autocast():
                    output = capsule_net(data.float())
                    loss = lossFunction(output, target.float())
                targetTensorForTest[batch_id * batchSizeTest:(batch_id + 1) * batchSizeTest, :] = target
                maskedTensorForTest[batch_id * batchSizeTest:(batch_id + 1) * batchSizeTest, :] = output
                test_loss += loss.item()
            correctExamplesNumTest = sum(np.argmax(maskedTensorForTest.data.cpu().numpy(), 1) == 
                                         np.argmax(targetTensorForTest.data.cpu().numpy(), 1))
            acc = correctExamplesNumTest / 33
        print("Test accuracy:", acc)
        print("Test Loss:", test_loss)
        if(acc > finalAcc[cv]):
            finalAcc[cv] = acc
print(finalAcc)
print("Acc = ", np.mean(finalAcc))
    
    
    
    