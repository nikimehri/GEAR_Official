import torch
import torch.nn as nn

from thirdparty.repdistiller.helper.loops import train_negrad


def negative_grad(model, data_loader, forget_loader, alpha, lr=0.01, epochs=10, quiet=False):
    """NegGrad baseline: jointly minimizes remain-set loss while maximizing
    (gradient-ascending) forget-set loss, blended by alpha - each epoch's
    combined loss is alpha*remain_loss - (1-alpha)*forget_loss."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        train_negrad(epoch, data_loader, forget_loader, model, loss_fn, optimizer,  alpha)
