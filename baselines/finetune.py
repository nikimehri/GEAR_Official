import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from baseline_utils import *
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate
from thirdparty.repdistiller.helper.pretrain import init

import copy


def finetune(model, data_loader, lr=0.01, epochs=10, quiet=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        train_vanilla(epoch, data_loader, model, loss_fn, optimizer)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser('Finetune unlearning baseline')
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.04)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset, testset, dataset = get_dataset('cifar100', args.data_root)
    num_classes, idx_to_class = set_num_classes('cifar100', dataset)

    total_forget = sum(1 for _, t in dataset if t == args.forget_class)
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, _, \
        _, _, _, _ = get_unlearn_loader(
            trainset, testset, args.forget_class, args.batch_size, num_forget=total_forget)

    final_forget_loader, final_remain_loader = get_forget_loader(testset, args.forget_class)
    _, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)

    model = load_model('resnet50', num_classes=num_classes, data_name='cifar100').to(device)
    model = load_model_state(model, args.checkpoint, map_location=device)

    finetune(model, train_remain_loader, lr=args.lr, epochs=args.num_epochs)

    _, test_acc = eval(model, test_loader, device=device)
    _, forget_acc = eval(model, final_forget_loader, device=device)
    _, remain_acc = eval(model, final_remain_loader, device=device)

    print(f'Seed {args.seed} -> test: {test_acc:.4f}  forget: {forget_acc:.4f}  remain: {remain_acc:.4f}')

    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'forget_class': args.forget_class,
            'data_root': args.data_root,
            'num_epochs': args.num_epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
        'results': {
            'test_acc': float(test_acc),
            'forget_acc': float(forget_acc),
            'remain_acc': float(remain_acc),
        },
    }

    out_path = f'finetune_results_seed{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {out_path}')

