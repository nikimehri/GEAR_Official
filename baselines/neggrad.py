import torch
import torch.nn as nn

from baseline_utils import *
from thirdparty.repdistiller.helper.loops import train_negrad


def negative_grad(model, data_loader, forget_loader, alpha, lr=0.01, epochs=10, quiet=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        train_negrad(epoch, data_loader, forget_loader, model, loss_fn, optimizer,  alpha)
