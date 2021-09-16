import os
import torch
from torch.autograd import grad
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

# config
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
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # .log_softmax(2) ## reshape x to (T,1,C),  1 is batch_size
        return x.reshape((T, 1, C))


# CTC loss - Pytorch: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

def get_random_sample():
    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(N, T, d)  # random X
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
    target = torch.randint(low=1, high=C, size=(
        sum(target_lengths),), dtype=torch.long)  # random Y, length of L

    return input, target, input_lengths, target_lengths


ctc_loss = nn.CTCLoss()

net = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def get_grads(x, y, x_len, y_len, detach=False):
    y_pred = net(x)
    loss = ctc_loss(y_pred, y, x_len, y_len)
    # loss = torch.norm(y_pred)
    net.zero_grad()
    loss_grads = grad(loss, net.parameters(),
                      create_graph=True)
    grads = {}
    for (name, _), g in zip(net.named_parameters(), loss_grads):
        if g is not None:
            grads[name] = g if not detach else g.cpu().detach()
    return grads


def get_grad_distance(grads1, grads2):
    keys = grads1.keys()
    g1 = torch.cat([torch.flatten(grads1[k]) for k in keys])
    g2 = torch.cat([torch.flatten(grads2[k]) for k in keys])
    return torch.norm(g1 - g2)


if __name__ == "__main__":
    # Compute grads
    x_gt, y_gt, x_len, y_len = get_random_sample()
    client_grads = get_grads(x_gt, y_gt, x_len, y_len, detach=True)

    x, y, x_len, y_len = get_random_sample()
    for _ in range(10):
        grads = get_grads(x, y, x_len, y_len)
        grad_dist = get_grad_distance(grads, client_grads)

        net.zero_grad()
        print(grad_dist)
        print(grad(grad_dist, net.parameters()))
