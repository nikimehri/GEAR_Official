"""
coun.py — COUN (Contrastive Unlearning via Nearest-Neighbour) baseline.

Loads a pretrained ResNet-50 on CIFAR-100 and runs the COUN update (verbatim
from the paper) across seeds [45, 46, 47, 48, 49, 50], then reports a
per-seed table and mean±std.  Hyperparameter sweep runs on seed=44 (not in
the eval set).

Run command:
    python baselines/coun.py \
        --forget_class 0 \
        --data_root ./data \
        --num_epochs 50 \
        [--checkpoint model_checkpoints/resnet50_cifar100_original_model_50_final_model_0.82.pth] \
        [--batch_size 64]
"""

import argparse
import copy
import json
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

# class_hierarchy.py lives at the repo root, one directory up from
# baselines/ - add it to sys.path the same way baseline_utils.py does, so
# this resolves regardless of the caller's working directory. Everything
# else in this file is deliberately self-contained; this is the one shared
# module it imports, since Retain Adjacent/Remote Accuracy should have one
# implementation, not a duplicated copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import class_hierarchy

# =============================================================================
# CoUn code from paper by Khalil et al., 2025
# =============================================================================

features = None
hook_fn = lambda module, _, output: globals().__setitem__('features', output)

def coun(model, layer, optimizer, retain_loader, transform, lambda_scale, temp):
    """From CoUn article by Khalil et al., 2025 section B.2 (PyTorch Code):
    Apply contrastive learning for unlearning using only retain data
    Args:
    model: the original model that needs to be unlearned
    layer: the penultimate layer for extracting embeddings
    optimizer: the optimizer to train the model
    retain_loader: the dataloader containing the retain data
    transform: the transformation to be applied to input images
    lambda_scale: a scaling constant for the CL loss
    temp: the temperature to be applied in the CL loss
    returns 'unlearned model' """
    # FIX 4: register hook once and remove it after the loop to prevent accumulation.
    handle = layer.register_forward_hook(hook_fn)  #Used to attach the hook at the penultimate layer
    try:
        for images, targets in retain_loader:
            batch_size = int(images.shape[0])
            images1, images2 = transform(images), transform(images) #Create two views of images
            outputs = model(images1) #Get model outputs and extract embeddings for images 1
            features1 = features.view(batch_size, -1)
            _ = model(images2) #Extract embeddings for images 2
            features2 = features.view(batch_size, -1)
            supervised_loss = nn.CrossEntropyLoss()(outputs, targets) #Supervised learning loss for images1
            # Single-line deviation from Khalil et al. 2025 verbatim code: device argument added so intra_mask is on the GPU.
            target = torch.arange(batch_size, device=images.device).unsqueeze(0) #Contrastive learning using the two views of the embeddings
            intra_mask = (torch.eq(target, target.T).float())
            cos_sim_ij = F.cosine_similarity(features1[:,None,:], features2[None,:,:], dim=-1)
            cos_sim_ij = torch.div(cos_sim_ij, temp)
            log_prob_ij = cos_sim_ij - torch.log((torch.exp(cos_sim_ij)).sum(1, keepdim=True))
            mean_log_prob_pos_ij = (intra_mask * log_prob_ij).sum(1) / intra_mask.sum(1)
            cos_sim_ji = F.cosine_similarity(features2[:,None,:], features1[None,:,:], dim=-1)
            cos_sim_ji = torch.div(cos_sim_ji, temp)
            log_prob_ji = cos_sim_ji - torch.log((torch.exp(cos_sim_ji)).sum(1, keepdim=True))
            mean_log_prob_pos_ji = (intra_mask * log_prob_ji).sum(1) / intra_mask.sum(1)
            contrastive_loss = -(mean_log_prob_pos_ij.mean() + mean_log_prob_pos_ji.mean())
            loss = supervised_loss + lambda_scale * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    finally:
        handle.remove()
    return model

# =============================================================================
# SimCLR augmentation transform
# FIX 1b: color augs run on raw [0,1] tensors; Normalize is the final step.
# The retain loader now yields un-normalized images so this ordering is valid.
# =============================================================================

_img_size = 32
_simclr_single = transforms.Compose([
    transforms.RandomResizedCrop(_img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

def simclr_transform(images):
    """Apply SimCLR augmentation independently to each image in a batch.
    Expects raw [0,1] tensors; normalizes as the last step.
    Transforms run on CPU; output is moved back to the input's device."""
    device = images.device
    cpu_images = images.cpu()
    transformed = torch.stack([_simclr_single(img) for img in cpu_images])
    return transformed.to(device)

# =============================================================================
# EVALUATION METRICS (copied from gear.py — kept self-contained so this
# baseline can run independently of the main pipeline)
# =============================================================================

def compute_mia(model, forget_loader, test_loader, device, n_splits=5):
    model.eval()

    def get_confidence_scores(loader):
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                scores.append(probs.cpu().numpy())
        return np.vstack(scores)

    test_scores = get_confidence_scores(test_loader)
    forget_scores = get_confidence_scores(forget_loader)

    n = min(len(forget_scores), len(test_scores))
    rng = np.random.default_rng(42)
    forget_scores = forget_scores[rng.choice(len(forget_scores), n, replace=False)]
    test_scores   = test_scores[rng.choice(len(test_scores),   n, replace=False)]

    X = np.concatenate([forget_scores, test_scores])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    clf = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        fold_accs.append(balanced_accuracy_score(y[test_idx], preds))

    return float(np.mean(fold_accs)), float(np.std(fold_accs))


def compute_loss_threshold_mia(model, member_loader, nonmember_loader, device):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    def collect_losses(loader):
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                per_sample = criterion(logits, y)
                losses.append(per_sample.cpu().numpy())
        return np.concatenate(losses)

    member_losses    = collect_losses(member_loader)
    nonmember_losses = collect_losses(nonmember_loader)

    losses = np.concatenate([member_losses, nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])

    mia_auc = roc_auc_score(labels, -losses)

    best_acc = 0.5
    for thresh in np.unique(losses):
        preds = (losses <= thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tpr = tp / max((labels == 1).sum(), 1)
        tnr = tn / max((labels == 0).sum(), 1)
        bal_acc = 0.5 * (tpr + tnr)
        if bal_acc > best_acc:
            best_acc = bal_acc

    return {'mia_auc': float(mia_auc), 'mia_acc': float(best_acc)}


# =============================================================================
# ACCURACY EVALUATION (copied from trainer.py — kept self-contained)
# =============================================================================

def eval_model(model, data_loader, device='cpu'):
    """Overall accuracy over a full loader pass."""
    model.eval()
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        if len(batch_y.shape) > 1:
            batch_y = batch_y.squeeze()
        if batch_y.dim() == 0:
            batch_y = batch_y.unsqueeze(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)
    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


# =============================================================================
# DATA LOADING (mirrors main.py exactly)
# =============================================================================

def get_cifar100(data_root):
    """Standard normalized CIFAR-100 train/test sets, used for evaluation."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    trainset_full = datasets.CIFAR100(root=data_root, train=True,  download=True, transform=train_transform)
    testset       = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
    return trainset_full, testset


# FIX 1a: raw [0,1] variant for the retain loader passed into coun().
# Color augs must run on [0,1] data; simclr_transform appends Normalize itself.
def get_cifar100_coun(data_root):
    """Raw [0,1] (un-normalized) CIFAR-100 train set for coun()'s retain
    loader - simclr_transform expects raw pixels and normalizes itself."""
    coun_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # → [0,1], NO normalize
    ])
    trainset_coun = datasets.CIFAR100(root=data_root, train=True, download=True,
                                      transform=coun_train_transform)
    return trainset_coun


def split_class_data(dataset, forget_class, num_forget):
    """Splits a dataset's indices into forget/remain/class_remain index
    lists (same idea as make_dataloaders.split_class_data)."""
    forget_index = []
    class_remain_index = []
    remain_index = []
    total = 0
    for i, (data, target) in enumerate(dataset):
        if target == forget_class and total < num_forget:
            forget_index.append(i)
            total += 1
        elif target == forget_class and total >= num_forget:
            class_remain_index.append(i)
            remain_index.append(i)
            total += 1
        else:
            remain_index.append(i)
    return forget_index, remain_index, class_remain_index


def get_forget_loader(dt, forget_class, batch_size=8):
    """Mirrors make_dataloaders.get_forget_loader exactly."""
    idx = []
    els_idx = []
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = DataLoader(dt, batch_size=batch_size, shuffle=False,
                               sampler=SubsetRandomSampler(idx), drop_last=True)
    remain_loader = DataLoader(dt, batch_size=batch_size, shuffle=False,
                               sampler=SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader


def build_loaders(trainset, testset, forget_class, batch_size, seed,
                  trainset_coun=None):
    """Build all loaders needed for coun() and evaluation, matching main.py.

    trainset_coun: if provided, the retain loader for coun() is built from this
    dataset (raw [0,1] tensors) using the same remain_index as trainset.  All
    other loaders use the normalized trainset / testset.
    """
    # Count forget samples (FORGET_PERCENTAGE = 1)
    total_forget_class = sum(1 for _, t in trainset if t == forget_class)
    num_forget = total_forget_class

    # Train forget / retain indices
    forget_index, remain_index, _ = split_class_data(trainset, forget_class, num_forget)

    train_forget_loader = DataLoader(trainset, batch_size=batch_size,
                                     sampler=SubsetRandomSampler(forget_index))

    # FIX 1c: retain loader for coun() uses raw [0,1] dataset when provided.
    if trainset_coun is not None:
        train_remain_loader = DataLoader(trainset_coun, batch_size=batch_size,
                                         sampler=SubsetRandomSampler(remain_index))
    else:
        train_remain_loader = DataLoader(trainset, batch_size=batch_size,
                                         sampler=SubsetRandomSampler(remain_index))

    # Test loaders (always use normalized testset)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    test_forget_loader, test_remain_loader = get_forget_loader(testset, forget_class, batch_size=8)

    return train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader


# =============================================================================
# DEVICE-AWARE LOADER WRAPPER
# =============================================================================

class DeviceLoader:
    """Wraps a DataLoader, moving every batch to `device` before yielding."""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for x, y in self.loader:
            yield x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.loader)


# =============================================================================
# PER-SEED RUNNER
# =============================================================================

def run_seed(seed, forget_class, data_root, checkpoint_path, num_epochs, batch_size, device,
             lambda_scale=1.0, temp=0.1):
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data — mirrors main.py split logic
    trainset_full, testset = get_cifar100(data_root)
    # FIX 1c: separate trainset with raw [0,1] images for the coun() retain loader.
    trainset_coun_full = get_cifar100_coun(data_root)

    val_fraction = 0.1
    val_size = int(len(trainset_full) * val_fraction)
    train_size = len(trainset_full) - val_size

    trainset, _ = random_split(
        trainset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    # Same indices for the coun variant (same seed → same split).
    trainset_coun, _ = random_split(
        trainset_coun_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader = \
        build_loaders(trainset, testset, forget_class, batch_size, seed,
                      trainset_coun=trainset_coun)

    # Load checkpoint (same pre-trained model for all seeds; seed controls splits)
    model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    # Optimizer and scheduler for coun()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    # COUN update — layer4 lives inside resnet_base for CustomResNet
    layer = model.resnet_base.layer4
    # Wrap loader so images/targets arrive on the correct device inside coun().
    retain_loader_device = DeviceLoader(train_remain_loader, device)
    print(f"Running COUN for {num_epochs} epoch(s) on retain set "
          f"({len(train_remain_loader)} batches/epoch) ...")
    for epoch in range(num_epochs):
        print(f"  epoch {epoch+1}/{num_epochs}")
        coun(model, layer, optimizer, retain_loader_device, simclr_transform,
             lambda_scale=lambda_scale, temp=temp)
        scheduler.step()

    model.eval()

    # Evaluation — mirrors gear.py evaluation block exactly
    _, forget_acc = eval_model(model, test_forget_loader, device)
    _, remain_acc = eval_model(model, test_remain_loader, device)
    # FIX 2: MIA members must be TRAINING-set forget samples (data the model saw),
    # not test forget samples.
    mia_mean, mia_std = compute_mia(model, train_forget_loader, test_loader, device)
    lt_mia = compute_loss_threshold_mia(model, train_forget_loader, test_forget_loader, device)

    # Retain Adjacent/Remote Accuracy (class-taxonomy-based - see
    # class_hierarchy.py). CoUn is CIFAR-100-only, so data_name is fixed.
    adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(
        'cifar100', forget_class, testset)
    retain_adjacent_acc, retain_remote_acc = class_hierarchy.compute_split_accuracy(
        model, testset, adjacent_indices, remote_indices, device)

    fa = forget_acc.item() if isinstance(forget_acc, torch.Tensor) else float(forget_acc)
    ra = remain_acc.item() if isinstance(remain_acc, torch.Tensor) else float(remain_acc)

    print(f"  forget_acc={fa:.4f}  remain_acc={ra:.4f}  "
          f"mia={mia_mean:.4f}±{mia_std:.4f}  "
          f"lt_mia_auc={lt_mia['mia_auc']:.4f}  lt_mia_acc={lt_mia['mia_acc']:.4f}")

    return {
        'seed': seed,
        'forget_acc': fa,
        'remain_acc': ra,
        'retain_adjacent_acc': retain_adjacent_acc,
        'retain_remote_acc': retain_remote_acc,
        'mia_mean': mia_mean,
        'mia_std': mia_std,
        'lt_mia_auc': lt_mia['mia_auc'],
        'lt_mia_acc': lt_mia['mia_acc'],
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    import sys
    sys.argv[1:] = [a.strip() for a in sys.argv[1:] if a.strip()]
    p = argparse.ArgumentParser(description='COUN baseline for CIFAR-100 ResNet-50')
    p.add_argument('--forget_class', type=int, default=0,
                   help='Class index to unlearn (default: 0)')
    p.add_argument('--data_root', type=str, default='./data',
                   help='Root directory for CIFAR-100 data')
    p.add_argument('--num_epochs', type=int, default=1,
                   help='Number of COUN epochs per seed (default: 1)')
    p.add_argument('--checkpoint', type=str,
                   default='model_checkpoints/resnet50_cifar100_original_model_50_final_model_0.82.pth',
                   help='Path to original (pre-unlearning) model checkpoint')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for retain loader and evaluation (default: 64)')
    p.add_argument('--gpu_id', type=int, default=0,
                   help='GPU id (default: 0; falls back to CPU if unavailable)')
    p.add_argument('--skip_sweep', action='store_true',
                   help='Skip hyperparameter sweep and use --lambda_scale / --temp directly')
    p.add_argument('--lambda_scale', type=float, default=1.0,
                   help='lambda_scale for coun() when --skip_sweep is set (default: 1.0)')
    p.add_argument('--temp', type=float, default=0.1,
                   help='temp for coun() when --skip_sweep is set (default: 0.1)')
    return p.parse_args()


def run_sweep(forget_class, data_root, checkpoint_path, num_epochs, batch_size, device):
    """Grid search over lambda_scale x temp on seed=44 (not in eval set).
    Returns (best_lambda, best_temp, sweep_rows, orig_remain_acc, remain_floor)."""
    lambda_candidates = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
    temp_candidates   = [0.05, 0.1, 0.2, 0.3]

    # FIX 3: compute original model's remain accuracy to set the floor.
    _, testset_ref = get_cifar100(data_root)
    _, test_remain_loader_ref = get_forget_loader(testset_ref, forget_class, batch_size=batch_size)
    orig_model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(orig_model, nn.DataParallel):
        orig_model = orig_model.module
    orig_model = copy.deepcopy(orig_model)
    orig_model.to(device)
    _, _orig_ra = eval_model(orig_model, test_remain_loader_ref, device)
    orig_remain_acc = _orig_ra.item() if isinstance(_orig_ra, torch.Tensor) else float(_orig_ra)
    remain_floor = 0.75 * orig_remain_acc
    del orig_model

    print(f"\n{'='*70}")
    print("  Hyperparameter sweep (seed=44)")
    print(f"  lambda_scale: {lambda_candidates}")
    print(f"  temp:         {temp_candidates}")
    print(f"  orig_remain_acc: {orig_remain_acc:.4f}  remain_floor (75%): {remain_floor:.4f}")
    print(f"{'='*70}")

    sweep_rows = []
    for lam in lambda_candidates:
        for tmp in temp_candidates:
            print(f"\n  sweep: lambda_scale={lam}  temp={tmp}")
            r = run_seed(
                seed=44,
                forget_class=forget_class,
                data_root=data_root,
                checkpoint_path=checkpoint_path,
                num_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
                lambda_scale=lam,
                temp=tmp,
            )
            sweep_rows.append({
                'lambda_scale': lam,
                'temp': tmp,
                'forget_acc': r['forget_acc'],
                'remain_acc': r['remain_acc'],
                'mia_mean':   r['mia_mean'],
                'lt_mia_auc': r['lt_mia_auc'],
                'lt_mia_acc': r['lt_mia_acc'],
            })

    # Print sweep results table
    print(f"\n{'='*80}")
    print(f"{'lambda':>8}  {'temp':>6}  {'ForgetAcc':>10}  {'RemainAcc':>10}  "
          f"{'MIA':>8}  {'LT-AUC':>8}")
    print(f"{'-'*80}")
    for row in sweep_rows:
        print(f"{row['lambda_scale']:>8.2f}  {row['temp']:>6.3f}  "
              f"{row['forget_acc']:>10.4f}  {row['remain_acc']:>10.4f}  "
              f"{row['mia_mean']:>8.4f}  {row['lt_mia_auc']:>8.4f}")
    print(f"{'='*80}")

    # FIX 3: select best only among configs that preserve remain_acc >= floor.
    valid = [r for r in sweep_rows if r['remain_acc'] >= remain_floor]
    if valid:
        best = min(valid, key=lambda x: (x['forget_acc'], -x['remain_acc']))
    else:
        print(f"\n  WARNING: no CoUn config cleared remain_floor={remain_floor:.4f}. "
              f"Falling back to highest remain_acc setting.")
        best = max(sweep_rows, key=lambda x: x['remain_acc'])

    print(f"\n  Selected: lambda_scale={best['lambda_scale']}  temp={best['temp']}"
          f"  (forget_acc={best['forget_acc']:.4f}  remain_acc={best['remain_acc']:.4f})")

    return best['lambda_scale'], best['temp'], sweep_rows, orig_remain_acc, remain_floor


def main():
    args = parse_args()

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    print(f"Forget class: {args.forget_class}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epochs per seed: {args.num_epochs}")

    # ── Hyperparameter selection ──────────────────────────────────────────────
    sweep_results = None
    orig_remain_acc = None
    remain_floor = None
    if args.skip_sweep:
        best_lambda = args.lambda_scale
        best_temp   = args.temp
        print(f"Skipping sweep — using lambda_scale={best_lambda}  temp={best_temp}")
    else:
        best_lambda, best_temp, sweep_results, orig_remain_acc, remain_floor = run_sweep(
            forget_class=args.forget_class,
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
        )

    # FIX 5: eval seeds [45..50], matching the seeds used for the main method.
    seeds = [45, 46, 47, 48, 49, 50]
    all_results = []

    for seed in seeds:
        result = run_seed(
            seed=seed,
            forget_class=args.forget_class,
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
            lambda_scale=best_lambda,
            temp=best_temp,
        )
        all_results.append(result)

    # Summary table
    keys = ['forget_acc', 'remain_acc', 'mia_mean', 'lt_mia_auc', 'lt_mia_acc']
    print(f"\n{'='*70}")
    print(f"{'Seed':>6}  {'ForgetAcc':>10}  {'RemainAcc':>10}  "
          f"{'MIA':>8}  {'LT-AUC':>8}  {'LT-ACC':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['seed']:>6}  {r['forget_acc']:>10.4f}  {r['remain_acc']:>10.4f}  "
              f"{r['mia_mean']:>8.4f}  {r['lt_mia_auc']:>8.4f}  {r['lt_mia_acc']:>8.4f}")

    print(f"{'-'*70}")
    means = {k: np.mean([r[k] for r in all_results]) for k in keys}
    stds  = {k: np.std( [r[k] for r in all_results]) for k in keys}
    print(f"{'mean±std':>6}  "
          f"{means['forget_acc']:.4f}±{stds['forget_acc']:.4f}  "
          f"{means['remain_acc']:.4f}±{stds['remain_acc']:.4f}  "
          f"{means['mia_mean']:.4f}±{stds['mia_mean']:.4f}  "
          f"{means['lt_mia_auc']:.4f}±{stds['lt_mia_auc']:.4f}  "
          f"{means['lt_mia_acc']:.4f}±{stds['lt_mia_acc']:.4f}")
    print(f"{'='*70}")

    # Save JSON
    output = {
        'config': {
            'forget_class': args.forget_class,
            'data_root': args.data_root,
            'checkpoint': args.checkpoint,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'seeds': seeds,
            'sweep_seed': 44,
            'lambda_scale': best_lambda,
            'temp': best_temp,
            'sweep_run': not args.skip_sweep,
            'orig_remain_acc': orig_remain_acc,
            'remain_floor': remain_floor,
        },
        'sweep_results': sweep_results,
        'per_seed': all_results,
        'summary': {
            k: {'mean': float(means[k]), 'std': float(stds[k])}
            for k in keys
        },
    }
    with open('coun_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to coun_results.json")


if __name__ == '__main__':
    main()
