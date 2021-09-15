import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import string
import random

import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

## config
T = 50
d = 30
C = 27     # num classes (including blank)
N = 1      # Batch size
L = 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(T*d, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, T*C)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape((T,1,C)) #.log_softmax(2) ## reshape x to (T,1,C),  1 is batch_size


### CTC loss - Pytorch: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(N, T, d).detach().requires_grad_()  # random X
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)  # random Y, length of L
ctc_loss = nn.CTCLoss()

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

grad = {}
grad["X_train"] = list()
grad["y_train"] = list()

for epoch in range(200):
    X_train = input
    y_train = target

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_pred = net(X_train)

    loss = ctc_loss(y_pred, y_train, input_lengths, target_lengths)

    print(loss.item())

    loss.backward()
    grad["X_train"].append(X_train.grad)
    grad["y_train"].append(y_train.grad)

    optimizer.step()

print('Done')

print(torch.sum(grad["X_train"][0]-grad["X_train"][-1]))