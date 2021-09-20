# Add noise, if grad_dist decrease, then x=x+noise, else continue

import os
import torch
from torch.autograd import grad
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,1)
        self.fc1.weight.data = torch.full((1,1), 0.5)
        self.fc1.bias.data = torch.full((1,), 1.)

    def forward(self, x):
        y_pred = self.fc1(x)
        return y_pred

net = Net()

def get_grads(x, y, detach=False):
    y_pred = net(x)
    # loss = ctc_loss(y_pred, y, x_len, y_len)
    loss = torch.sum((y_pred-y)**2)
    net.zero_grad()
    loss_grads = grad(loss, net.parameters(), create_graph=True)
    grads = {}
    for (name, _), g in zip(net.named_parameters(), loss_grads):
        if g is not None:
            grads[name] = g if not detach else g.cpu().detach()
    return grads

def get_grad_manually(x,y,w,b):
    grads= {}

    grads["dl/dw"] = 2 * x @ (x @ w + b - y)
    grads["dl/db"] = 2 * (x @ w + b - y)

    grads["dl/dx"] = 2 * w @ (x@w+b-y)
    grads["dl/dy"] = -2 * (x @ w + b - y)

    return grads

def get_grad_distance(grads1, grads2, keys=None):
    if keys is None:
        keys = grads1.keys()
    g1 = torch.cat([torch.flatten(grads1[k]) for k in keys])
    g2 = torch.cat([torch.flatten(grads2[k]) for k in keys])
    return torch.sum ((g1 - g2)**2)


def add_noise_x(x):
    return x + torch.normal(mean=0, std=0.1, size=(1,))


dist_list = []

if __name__ == '__main__':
    x_gt = torch.tensor([3.], requires_grad=True)
    y_gt = torch.tensor([3 * (.45) + 1], requires_grad=True)

    client_grads = get_grads(x_gt, y_gt, detach=True)  ## device
    client_grads_manually = get_grad_manually(x_gt,y_gt, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'])

    assert (client_grads["fc1.weight"] == client_grads_manually["dl/dw"] \
            and client_grads["fc1.bias"] == client_grads_manually["dl/db"])

    x = torch.tensor([2.], requires_grad=True)
    y = y_gt # fix value of y

    for _ in range(300):
        grads = get_grads(x, y)
        grads_manually = get_grad_manually(x,y, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'])

        assert grads["fc1.weight"] == grads_manually["dl/dw"]
        assert grads["fc1.bias"] == grads_manually["dl/db"]


        grad_dist = get_grad_distance(grads, client_grads)  # client_grads: constant
        grad_dist_manually = get_grad_distance(grads_manually, client_grads_manually, keys = ["dl/dw", "dl/db"])


        x_new = add_noise_x(x)
        grads_new = get_grads(x_new,y)
        grads_manually_new = get_grad_manually(x_new, y, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'])
        assert (grads_new["fc1.weight"] == grads_manually_new["dl/dw"] \
                and grads_new["fc1.bias"] == grads_manually_new["dl/db"])

        grad_dist_new = get_grad_distance(grads_manually_new, client_grads_manually, keys = ["dl/dw", "dl/db"])  ### client_grads: constant

        if grad_dist_new.item() < grad_dist_manually.item():
            x=x_new
            dist_list.append(grad_dist_manually.item())



        net.zero_grad()
        print("grad_dist: ", grad_dist)
        print("grad_dist_manually: ", grad_dist_manually)

        # make_dot(grad_dist, params=dict(net.named_parameters()), show_saved=True).render("norm_loss", format="png")


    sns.set_theme(style="darkgrid")

    x_ = list(range(len(dist_list)))
    y_ = dist_list

    sns.lineplot(x_, y_)
    plt.show()

    print("final x:", x)

