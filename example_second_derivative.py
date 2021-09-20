import torch
from torch.autograd import grad
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc1.weight.data = torch.full((1, 1), 0.5)
        self.fc1.bias.data = torch.full((1,), 1.)

    def forward(self, x):
        y_pred = self.fc1(x)
        return y_pred


net = Net()


def get_grads(x, y, detach=False):
    y_pred = net(x)
    # loss = ctc_loss(y_pred, y, x_len, y_len)
    loss = torch.sum((y_pred - y) ** 2)
    net.zero_grad()
    loss_grads = grad(loss, net.parameters(), create_graph=True)
    grads = {}
    for (name, _), g in zip(net.named_parameters(), loss_grads):
        if g is not None:
            grads[name] = g if not detach else g.cpu().detach()
    return grads


def get_grad_manually(x, y, w, b):
    grads = {}

    grads["dl/dw"] = 2 * x @ (x @ w + b - y)
    grads["dl/db"] = 2 * (x @ w + b - y)

    grads["dl/dx"] = 2 * w @ (x @ w + b - y)
    grads["dl/dy"] = -2 * (x @ w + b - y)

    return grads


def get_2nd_grad_manually(x, y, w, b, dw_gt, db_gt):
    """

    :param x:
    :param y:
    :param w:
    :param b:
    :param dw_gt: dL/dW_gt
    :param db_gt: dL/db_gt
    :return:
    """
    grads = {}

    grads["dl/dw"] = 2 * x @ (x @ w + b - y)
    grads["dl/db"] = 2 * (x @ w + b - y)

    grads["d2l/dwdx"] = 2 * (grads["dl/dw"] - dw_gt) * 2 * (2 * x @ w + b - y)
    grads["d2l/dbdx"] = 2 * (grads["dl/db"] - db_gt) * 2 * w
    grads["d/dx"] = grads["d2l/dwdx"] + grads["d2l/dbdx"]

    grads["d2l/dwdy"] = 2 * (grads["dl/dw"] - dw_gt) * (-2 * x)
    grads["d2l/dbdy"] = 2 * (grads["dl/db"] - db_gt) * (-2)
    grads["d/dy"] = grads["d2l/dwdy"] + grads["d2l/dbdy"]

    return grads["d/dx"], grads["d/dy"]


def get_grad_distance(grads1, grads2, keys=None):
    if keys is None:
        keys = grads1.keys()
    g1 = torch.cat([torch.flatten(grads1[k]) for k in keys])
    g2 = torch.cat([torch.flatten(grads2[k]) for k in keys])
    return torch.sum((g1 - g2) ** 2)



if __name__ == '__main__':
    grad_dist_list = []

    x_gt = torch.tensor([3.], requires_grad=True)
    y_gt = torch.tensor([2.35], requires_grad=True)

    client_grads = get_grads(x_gt, y_gt, detach=True)
    client_grads_manually = get_grad_manually(x_gt, y_gt, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'])

    assert (client_grads["fc1.weight"] == client_grads_manually["dl/dw"] \
            and client_grads["fc1.bias"] == client_grads_manually["dl/db"])

    x = torch.tensor([2.8], requires_grad=True)
    y = torch.tensor([2.35], requires_grad=True)  # fix value of y

    for _ in range(10):
        ## get gradient dl/dw, dl/dx
        grads = get_grads(x, y)
        grads_manually = get_grad_manually(x, y, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'])
        assert grads["fc1.weight"] == grads_manually["dl/dw"]
        assert grads["fc1.bias"] == grads_manually["dl/db"]

        ## calc gradient distance
        grad_dist = get_grad_distance(grads, client_grads)  # client_grads: constant
        grad_dist_manually = get_grad_distance(grads_manually, client_grads_manually, keys=["dl/dw", "dl/db"])

        print("grad_dist: ", grad_dist)
        print("grad_dist_manually: ", grad_dist_manually)

        ## keep track of grad_dists
        grad_dist_list.append(grad_dist.item())


        ## second derivative w.r.t. x, y
        grad_2nd = grad(grad_dist, (x, y))
        grad_2nd_manually = get_2nd_grad_manually(x, y, net.state_dict()['fc1.weight'], net.state_dict()['fc1.bias'],
                                                client_grads_manually["dl/dw"], client_grads_manually["dl/db"])
        print("grad_2nd_manually : ", grad_2nd_manually)
        print("grad_2nd : ", grad_2nd)


        ## Update x, y
        with torch.no_grad():
            x -= grad_2nd_manually[0].item() * 0000.1
            y -= grad_2nd_manually[1].item() * 0000.1


        # make_dot(grad_dist, params=dict(net.named_parameters()), show_saved=True).render("norm_loss", format="png")

        print("x: ", x)
        print("y: ", y)

        net.zero_grad()

        print("----\n")


    ## plot grad_dist_list over epochs
    sns.set_theme(style="darkgrid")
    x_ = list(range(len(grad_dist_list)))
    y_ = grad_dist_list
    sns.lineplot(x_, y_)
    plt.show()
    print("final x, y", x, y)
    print("true x, y:", x_gt, y_gt)
