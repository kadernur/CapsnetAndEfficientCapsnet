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
USE_CUDA = False

warnings.filterwarnings("ignore")
plt.ion()


class TrainDataset(torchdata.Dataset):
    def __init__(self, fileList, labels):
        self.fileList = fileList
        self.labels = labels

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index: int):
        filename = self.fileList[index]
        img = Image.open(filename)
        img = np.array(img.resize((28, 28), Image.ANTIALIAS))
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


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              ),nn.ReLU())
        self.norm1 = nn.BatchNorm2d()
        
        self.conv2 = nn.Sequential(nn.Conv2d(
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              ),nn.ReLU())
        self.norm2 = nn.BatchNorm2d(10)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              ),nn.ReLU())
        self.norm3 = nn.BatchNorm2d(10)
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              ),nn.ReLU())
        self.norm4 = nn.BatchNorm2d(10)

    def forward(self, x):
        
      feature_maps = self.norm1(self.conv1(x))
      feature_maps = self.norm2(self.conv2(feature_maps))
      feature_maps = self.norm3(self.conv3(feature_maps))
      return self.norm4(self.conv4(feature_maps))

       


     
     

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=15, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(
            1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(
                    3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 15, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(15))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(
            dim=0, index=max_length_indices.squeeze(1).data)

        reconstructions = self.reconstraction_layers(
            (x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)

        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(
            self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(
            reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

def build_dataset():
    fileList = []
    labels = []
    for i in range(1, 16):
        files = glob.glob('./data/subject'+str(i).zfill(2)+"*")
        for fname in files:
            fileList.append(fname)
            labels.append(i - 1)
    return fileList, np.array(labels)

batch_size = 22
batchSizeTest = int(batch_size / 2)
n_epochs = 200
finalAcc = np.zeros(5)
fileLinks, labels = build_dataset()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5)
for cv, (train_index, test_index) in enumerate(skf.split(fileLinks, labels)):

    mnist = Yale(batch_size, batchSizeTest, train_index, test_index)
    capsule_net = CapsNet()
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    optimizer = Adam(capsule_net.parameters(), lr=0.0001)

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
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
                output, reconstructions, masked = capsule_net(data.float())
                loss = capsule_net.loss(data, output, target.float(), reconstructions)
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=None, parameters=capsule_net.parameters(), create_graph=second_order)
            targetTensor[batch_id * batch_size:(batch_id + 1) * batch_size, :] = target
            maskedTensor[batch_id * batch_size:(batch_id + 1) * batch_size, :] = masked
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
                    output, reconstructions, masked = capsule_net(data.float())
                    loss = capsule_net.loss(data, output, target.float(), reconstructions)
                targetTensorForTest[batch_id * batchSizeTest:(batch_id + 1) * batchSizeTest, :] = target
                maskedTensorForTest[batch_id * batchSizeTest:(batch_id + 1) * batchSizeTest, :] = masked
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
