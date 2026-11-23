"""Nearest-class-centroid (NCC) analysis of the forget set, for comparing the
original / retrain / unlearned models on the same split GEAR is evaluated on.

Matches GEAR's setup: same seeded train/val split as main.py, same layer
(--target_layer), same feature prep (GAP + L2-normalise via gear.prepare_features),
centroids from the retain *train* set, evaluation on the *val* forget/remain
sets (what --specific_settings reports).

Two centroid variants per model:
  retain-only : centroids for classes != forget_class. The forget class has no
                centroid, so NCC forget acc is 0 by construction; the useful
                numbers are the mean max-cosine (GEAR's entanglement score e_i)
                and where forget samples land.
  full        : forget-class centroid built from train-forget features too.
                NCC forget/remain acc then say how separable the forget class
                still is in feature space.

Usage:
  python ncc_eval.py --seed 42 --target_layer 9 \
      --model original=model_checkpoints/AllCNN_cifar10_original_model_1.pth \
      --model retrain=model_checkpoints/AllCNN_cifar10_retrain_model_class_0_1.pth \
      --model bw1=bw_test_1.0.pth --model bw0=bw_test_0.pth
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from gear import get_intermediate_features, prepare_features
from make_dataloaders import get_dataset
from utils import seed_torch


def class_indices(ds, forget_class):
    fi, ri = [], []
    for i in range(len(ds)):
        (fi if ds[i][1] == forget_class else ri).append(i)
    return fi, ri


@torch.no_grad()
def extract(model, ds, idx, layer, device, bs):
    model.eval()
    feats, labels, logits = [], [], []
    for x, y in DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False):
        x = x.to(device)
        out = model(x)
        feats.append(prepare_features(get_intermediate_features(model, x, layer)).cpu())
        logits.append(out.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels), torch.cat(logits)


def centroids(feats, labels, classes):
    c = torch.stack([feats[labels == k].mean(0) for k in classes])
    return F.normalize(c, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', required=True, metavar='NAME=PATH')
    p.add_argument('--data_name', default='cifar10')
    p.add_argument('--model_name', default='AllCNN')
    p.add_argument('--dataset_dir', default='./data')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--forget_class', type=int, default=0)
    p.add_argument('--target_layer', default='9')
    p.add_argument('--batch_size', type=int, default=256)
    a = p.parse_args()

    seed_torch(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # Same split as main.py.
    trainset_full, _, dataset = get_dataset(a.data_name, a.dataset_dir, model_name=a.model_name)
    val_size = int(len(trainset_full) * a.val_fraction)
    gen = torch.Generator().manual_seed(a.seed)
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [len(trainset_full) - val_size, val_size], generator=gen)

    tr_f, tr_r = class_indices(trainset, a.forget_class)
    va_f, va_r = class_indices(valset, a.forget_class)
    print(f'device={device} | train forget/retain={len(tr_f)}/{len(tr_r)} | '
          f'val forget/retain={len(va_f)}/{len(va_r)} | layer={a.target_layer}')

    fc = a.forget_class
    print(f'\n{"model":10s} {"NCCfull F":>9s} {"NCCfull R":>9s} {"e_i(F)":>7s} {"e_i(R)":>7s} '
          f'{"F->top retain cls":>18s} {"softmax F":>9s}')
    for spec in a.model:
        name, path = spec.split('=', 1)
        model = torch.load(path, map_location=device, weights_only=False)
        model = (model.module if hasattr(model, 'module') else model).to(device)  # drop CUDA DataParallel wrapper

        trR_f, trR_y, _ = extract(model, trainset, tr_r, a.target_layer, device, a.batch_size)
        trF_f, trF_y, _ = extract(model, trainset, tr_f, a.target_layer, device, a.batch_size)
        vaF_f, vaF_y, vaF_l = extract(model, valset, va_f, a.target_layer, device, a.batch_size)
        vaR_f, vaR_y, vaR_l = extract(model, valset, va_r, a.target_layer, device, a.batch_size)

        classes = sorted(set(trR_y.tolist()) | {fc})
        retain_classes = [k for k in classes if k != fc]

        # retain-only: entanglement of forget samples with retain centroids
        cR = centroids(trR_f, trR_y, retain_classes)
        simF = vaF_f @ cR.T
        simR = vaR_f @ cR.T
        e_F, e_R = simF.max(1).values.mean().item(), simR.max(1).values.mean().item()
        top = torch.tensor(retain_classes)[simF.argmax(1)]
        top_cls = torch.mode(top).values.item()
        top_frac = (top == top_cls).float().mean().item()

        # full: add forget-class centroid from train-forget features
        cAll = centroids(torch.cat([trR_f, trF_f]), torch.cat([trR_y, trF_y]), classes)
        cls_t = torch.tensor(classes)
        ncc_F = (cls_t[(vaF_f @ cAll.T).argmax(1)] == fc).float().mean().item()
        ncc_R = (cls_t[(vaR_f @ cAll.T).argmax(1)] == vaR_y).float().mean().item()

        sm_F = (vaF_l.argmax(1) == fc).float().mean().item()
        print(f'{name:10s} {ncc_F:9.4f} {ncc_R:9.4f} {e_F:7.4f} {e_R:7.4f} '
              f'{f"{top_cls} ({top_frac:.2f})":>18s} {sm_F:9.4f}')


if __name__ == '__main__':
    main()
