"""Linear-probe evaluation for GEAR checkpoints (Original/Retrain/GEAR/any
baseline), following Gao et al., AISTATS 2026, "An Illusion of Unlearning?".

For a single saved checkpoint, measures how much forget-class information is
still linearly recoverable from its FROZEN penultimate features. Each
checkpoint gets its own fresh probe, trained on its own extracted features -
probes are never reused across checkpoints.

Design notes (decisions made explicit after inspecting the repo, rather than
assumed):

- Penultimate features are extracted via each model's existing
  `get_embedding(x)` method (AllCNN/CustomResNet/VGG/ViT/TextTransformer all
  define it, already handling DataParallel-unwrapping conventions
  identically across the codebase - see embeddings.py for the established
  pattern this follows), not a hand-rolled forward hook or a classifier ->
  Identity swap. This is the input to the final classifier layer for every
  architecture here, exactly matching this script's requirement.

- Checkpoints are loaded via plain `torch.load` - every checkpoint this repo
  produces (trainer.py's original/retrain, gear.py's unlearned model,
  baseline_main.py's --save_checkpoints output) is a full pickled model
  object, optionally nn.DataParallel-wrapped, never a bare state_dict. No
  --arch-driven model reconstruction is needed or attempted; --arch/--dataset
  are only used to resolve this checkpoint's test-time transform pipeline
  (via make_dataloaders.get_dataset) and expected classifier in_features
  (for the feature-dim sanity assertion).

- The train/val split EXACTLY reproduces main.py's actual split: a plain
  (non-stratified) torch.utils.data.random_split over the full training set,
  seeded by --seed, with the same default val_fraction=0.1 params.py uses.
  This was a deliberate choice (confirmed with the user) over a "true"
  stratified split, so the probe's val set is identical to whatever main.py
  used when training/unlearning this exact checkpoint - not an independently
  re-randomized split that happens to also be seeded.

- All three splits use the TEST-TIME transform (no augmentation), including
  train - built by reusing `testset.transform` (already constructed by
  get_dataset for this exact data_name/arch pair) on a second train=True
  dataset instance, rather than duplicating per-dataset normalization
  constants here. The probe train split is the SAME images main.py's
  train_size portion of random_split selected (same indices, same seed),
  just re-fetched through the test-time transform instead of the augmented
  one; it covers all classes (including the forget class, with its TRUE
  label) because no forget/remain split is ever applied before extraction.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

# Repo root is one directory up from probing/ - add it to sys.path so this
# resolves regardless of the caller's working directory, same pattern
# baselines/*.py already uses for class_hierarchy.py/ain_metric.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from make_dataloaders import get_dataset  # noqa: E402
from baselines.baseline_utils import set_num_classes  # noqa: E402


# =============================================================================
# DETERMINISM
# =============================================================================

def set_determinism(seed: int) -> None:
    """Seeds every RNG this script touches: torch, numpy, and the stdlib
    random module (torchvision transforms occasionally fall back to it)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Loads a full pickled model object (trainer.py/gear.py/baseline_main.py
    --save_checkpoints convention), unwraps nn.DataParallel if present, moves
    to device, and sets eval() mode. Raises a clear error if the checkpoint
    isn't a usable nn.Module (e.g. a bare state_dict) - this script doesn't
    support rebuilding a model from a state_dict via --arch, since no
    checkpoint in this repo is ever saved that way for a final result."""
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.DataParallel):
        obj = obj.module
    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"'{checkpoint_path}' did not load as an nn.Module (got {type(obj)}). "
            f"This script expects full pickled model-object checkpoints (the "
            f"convention trainer.py/gear.py/baseline_main.py --save_checkpoints "
            f"all use), not a bare state_dict."
        )
    if not hasattr(obj, 'get_embedding'):
        raise AttributeError(
            f"Loaded model ({type(obj).__name__}) has no get_embedding() method - "
            f"every architecture in models.py/text_models.py defines one; this "
            f"checkpoint may be from an unsupported/custom model class."
        )
    obj.to(device)
    obj.eval()
    return obj


_METHOD_FROM_FILENAME = re.compile(r'^[^_]+_[^_]+_(.+)_forget\d+_seed\d+$')


def infer_method_name(checkpoint_path: str, method_override: Optional[str] = None) -> str:
    """Resolves the method name for a checkpoint's result JSON. Prefers an
    explicit override; otherwise tries to parse it from the filename,
    recognizing trainer.py's original/retrain convention and
    baseline_main.py --save_checkpoints' convention
    ({arch}_{dataset}_{method}_forget{N}_seed{M}.pth). GEAR's own checkpoints
    are saved under a free-form --name with no fixed convention, so for those
    (and anything else unrecognized) this falls back to the bare filename
    stem - loudly, via a printed warning, never a silent guess."""
    if method_override:
        return method_override
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if '_original_model_' in stem:
        return 'original'
    if '_retrain_model_class_' in stem:
        return 'retrain'
    m = _METHOD_FROM_FILENAME.match(stem)
    if m:
        return m.group(1)
    print(f"[linear_probe] WARNING: could not infer a method name from checkpoint "
          f"filename '{stem}' - pass --method explicitly for a reliable label. "
          f"Falling back to the raw filename stem.")
    return stem


# =============================================================================
# DATA: splits that exactly reproduce main.py's protocol, test-time
# transform only, forget class included with true labels in train.
# =============================================================================

def _build_train_with_transform(data_name: str, data_dir: str, transform):
    """Re-fetches the SAME underlying training images get_dataset already
    built (same download, same on-disk order), but with `transform` swapped
    in - used to apply the test-time transform (taken from the already-
    constructed testset.transform) to the training split, instead of
    duplicating get_dataset's per-dataset normalization-constant logic here.
    Only the dataset classes/paths get_dataset itself already knows how to
    build are supported; everything else raises rather than guessing."""
    if data_name == 'cifar10':
        return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif data_name == 'cifar100':
        return datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    elif data_name == 'tinyimagenet':
        root = os.path.join(data_dir, 'tiny-imagenet-200')
        return datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    else:
        raise ValueError(
            f"linear_probe.py doesn't know how to rebuild a transform-swapped "
            f"train set for data_name={data_name!r}. Add a branch here if this "
            f"dataset needs probing."
        )


@dataclass
class ProbeSplits:
    train: Subset
    val: Subset
    test: object
    num_classes: int
    train_indices: list
    val_indices: list


def build_splits(data_name: str, arch: str, data_dir: str, seed: int,
                  val_fraction: float = 0.1) -> ProbeSplits:
    """Reproduces main.py's exact train/val split (plain, non-stratified
    random_split, seeded) over the full training set, then swaps every split
    to the test-time transform. See this module's docstring for the full
    reasoning."""
    trainset_full, testset, dataset_ref = get_dataset(data_name, data_dir, model_name=arch)
    num_classes, _ = set_num_classes(data_name, dataset_ref)

    val_size = int(len(trainset_full) * val_fraction)
    train_size = len(trainset_full) - val_size
    split_generator = torch.Generator().manual_seed(seed)
    train_subset_aug, val_subset_aug = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=split_generator
    )
    train_indices = list(train_subset_aug.indices)
    val_indices = list(val_subset_aug.indices)

    train_notransform = _build_train_with_transform(data_name, data_dir, testset.transform)
    probe_train = Subset(train_notransform, train_indices)
    probe_val = Subset(train_notransform, val_indices)

    return ProbeSplits(
        train=probe_train, val=probe_val, test=testset,
        num_classes=num_classes, train_indices=train_indices, val_indices=val_indices,
    )


# =============================================================================
# FEATURE EXTRACTION / CACHING
# =============================================================================

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    """Extracts frozen penultimate features via model.get_embedding(x).
    model must already be .eval() and on `device` - this function asserts
    that rather than setting it, so callers can't accidentally extract under
    train-mode BatchNorm/Dropout behavior. Runs entirely under
    @torch.no_grad(). Returns (features [N, D] float32, labels [N] int64),
    both on CPU."""
    assert not model.training, "extract_features requires model.eval() - BatchNorm must use running stats."
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        f = model.get_embedding(x)
        feats.append(f.detach().float().cpu())
        labels.append(y.detach().long().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def cache_path_for(cache_dir: str, checkpoint_path: str, split_name: str) -> str:
    """Cache key is the checkpoint's absolute path (so identically-named
    checkpoints in different directories never collide) plus the split name.
    A short hash keeps filenames bounded while staying human-readable via
    the retained filename stem."""
    key = hashlib.sha256(os.path.abspath(checkpoint_path).encode()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join(cache_dir, f"{stem}_{key}_{split_name}.pt")


def get_or_extract_features(model: nn.Module, loader: DataLoader, device: torch.device,
                             cache_dir: str, checkpoint_path: str, split_name: str):
    """Returns (features, labels) for this (checkpoint, split), using an
    fp16-on-disk cache keyed by checkpoint path + split so reruns skip
    extraction entirely. Cached features are upcast back to fp32 on load,
    matching extract_features' own return dtype so callers never need to
    know whether a result came from cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path_for(cache_dir, checkpoint_path, split_name)
    if os.path.exists(path):
        cached = torch.load(path, map_location='cpu', weights_only=True)
        return cached['features'].float(), cached['labels'].long()

    features, labels = extract_features(model, loader, device)
    torch.save({'features': features.half(), 'labels': labels}, path)
    return features, labels


# =============================================================================
# STANDARDIZATION
# =============================================================================

def compute_standardizer(train_features: torch.Tensor, eps: float = 1e-6):
    """Per-feature mean/std from the TRAIN split only - never val/test."""
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    return mean, std + eps


def standardize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


# =============================================================================
# PROBE TRAINING / EVALUATION
# =============================================================================

def train_probe(train_features: torch.Tensor, train_labels: torch.Tensor, num_classes: int,
                 weight_decay: float, device: torch.device, seed: int,
                 epochs: int = 50, lr: float = 1e-3, batch_size: int = 1024) -> nn.Linear:
    """Trains a fresh nn.Linear(feat_dim, num_classes) probe with
    cross-entropy on train_features/train_labels. Fixed recipe: AdamW, the
    given lr/batch_size/epochs, a frozen weight_decay (see select_recipe
    below for how that value is chosen). Deterministic given `seed`."""
    set_determinism(seed)
    gen = torch.Generator().manual_seed(seed)

    feat_dim = train_features.shape[1]
    probe = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)

    probe.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
    probe.eval()
    return probe


@torch.no_grad()
def evaluate_probe(probe: nn.Linear, features: torch.Tensor, labels: torch.Tensor,
                    forget_class: int, device: torch.device) -> dict:
    """Returns forget_class_probe_acc (headline metric), retain_class_probe_acc,
    overall_probe_acc, and the per-group sample counts."""
    probe.eval()
    preds = probe(features.to(device)).argmax(dim=1).cpu()
    correct = (preds == labels)

    forget_mask = labels == forget_class
    retain_mask = ~forget_mask

    def _safe_mean(mask):
        return correct[mask].float().mean().item() if mask.any() else float('nan')

    return {
        'overall_probe_acc': correct.float().mean().item(),
        'forget_class_probe_acc': _safe_mean(forget_mask),
        'retain_class_probe_acc': _safe_mean(retain_mask),
        'n_test_forget': int(forget_mask.sum()),
        'n_test_retain': int(retain_mask.sum()),
        'n_test_total': int(labels.shape[0]),
    }


@torch.no_grad()
def ncc_accuracy(train_features: torch.Tensor, train_labels: torch.Tensor,
                  test_features: torch.Tensor, test_labels: torch.Tensor,
                  num_classes: int, forget_class: int) -> Optional[float]:
    """Nearest-class-center accuracy on the forget class only: class means
    come from TRAIN features; test samples are assigned to their nearest
    (Euclidean) class mean. Returns None if the forget class has no train
    samples (center undefined) or no test samples (nothing to score)."""
    if not (train_labels == forget_class).any() or not (test_labels == forget_class).any():
        return None

    feat_dim = train_features.shape[1]
    centers = torch.zeros(num_classes, feat_dim)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.any():
            centers[c] = train_features[mask].mean(dim=0)
        else:
            centers[c] = float('inf')  # never the nearest center for any point

    dists = torch.cdist(test_features, centers)
    preds = dists.argmin(dim=1)

    forget_mask = test_labels == forget_class
    return (preds[forget_mask] == forget_class).float().mean().item()


# =============================================================================
# RECIPE SELECTION (frozen once, reused by every later probe run)
# =============================================================================

WEIGHT_DECAY_GRID = [0.0, 1e-4, 1e-3, 1e-2]


def select_recipe(train_features, train_labels, val_features, val_labels,
                   num_classes: int, device: torch.device, seed: int,
                   probe_kwargs: Optional[dict] = None) -> dict:
    """Sweeps WEIGHT_DECAY_GRID, training a fresh probe for each candidate on
    train_features and scoring overall VALIDATION accuracy. Returns
    {'weight_decay': <best>, 'val_acc_by_weight_decay': {...}} - meant to be
    run ONCE on a config's Original checkpoint, then written to disk and
    reused by every later probe run for that config (never re-tuned per
    method).

    probe_kwargs, when given, is forwarded to every train_probe() call (e.g.
    {'batch_size': 32, 'epochs': 20} for a small/synthetic sweep where the
    production defaults - sized for real CIFAR-scale extraction - would
    collapse to too few optimizer steps to converge). Omitted (the default)
    for every real call site in this module, so the production recipe sweep
    always uses train_probe's own frozen defaults, identical to the ones
    every later evaluation run will use."""
    probe_kwargs = probe_kwargs or {}
    results = {}
    for wd in WEIGHT_DECAY_GRID:
        probe = train_probe(train_features, train_labels, num_classes, wd, device, seed, **probe_kwargs)
        val_preds = probe(val_features.to(device)).argmax(dim=1).cpu()
        val_acc = (val_preds == val_labels).float().mean().item()
        results[wd] = val_acc
        print(f"[select_recipe] weight_decay={wd:g} -> val_acc={val_acc:.4f}")

    best_wd = max(results, key=results.get)
    print(f"[select_recipe] Selected weight_decay={best_wd:g} (val_acc={results[best_wd]:.4f})")
    return {
        'weight_decay': best_wd,
        'val_acc_by_weight_decay': {str(k): v for k, v in results.items()},
    }


def load_recipe(recipe_json: str) -> dict:
    if not os.path.exists(recipe_json):
        raise FileNotFoundError(
            f"--recipe_json '{recipe_json}' doesn't exist yet. Run this script with "
            f"--select_recipe against the Original checkpoint for this config first - "
            f"every later probe run reads the frozen weight_decay from this file and "
            f"must never re-tune it per method."
        )
    with open(recipe_json, 'r') as f:
        return json.load(f)


# =============================================================================
# CLI
# =============================================================================

def _classifier_in_features(model: nn.Module, arch: str) -> int:
    """Looks up the real in_features of model's final classifier layer, per
    architecture - used only for the feature-dim sanity assertion, not for
    any extraction logic (get_embedding already returns the right thing on
    its own)."""
    if arch in ('resnet', 'resnet50', 'resnet18'):
        return model.classifier[1].in_features  # Sequential(Dropout, Linear)
    if arch in ('vgg16', 'vgg16_bn'):
        return model.vgg.classifier[6].in_features
    if arch == 'vit':
        return model.vit.head.in_features
    if arch == 'distilbert':
        return model.classifier.in_features
    if arch == 'AllCNN' or arch == 'allcnn':
        return model.classifier[0].in_features
    raise ValueError(f"Don't know how to find the classifier in_features for arch={arch!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Linear-probe evaluation for a GEAR checkpoint")
    p.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint to probe")
    p.add_argument('--arch', type=str, required=True,
                   choices=['AllCNN', 'allcnn', 'resnet', 'resnet50', 'resnet18', 'vgg16', 'vgg16_bn', 'vit', 'distilbert'],
                   help="Architecture of the checkpoint (resolves its test-time transform/classifier shape - "
                        "the checkpoint itself is always loaded as a full model object, --arch is never used "
                        "to reconstruct one from a state_dict)")
    p.add_argument('--dataset', type=str, required=True,
                   choices=['cifar10', 'cifar100', 'tinyimagenet'],
                   help="Dataset this checkpoint was trained/unlearned on")
    p.add_argument('--forget_class', type=int, required=True)
    p.add_argument('--seed', type=int, required=True,
                   help="Seeds the probe's own training stochasticity AND reproduces main.py's "
                        "exact (seed-dependent) train/val split for this checkpoint - must match "
                        "the --seed the checkpoint was actually produced with.")
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--cache_dir', type=str, default='./probing/feature_cache')
    p.add_argument('--output_json', type=str, required=True)
    p.add_argument('--recipe_json', type=str, required=True,
                   help="Where the frozen weight_decay recipe is read from (normal mode) or "
                        "written to (--select_recipe mode)")
    p.add_argument('--method', type=str, default=None,
                   help="Method name for the result JSON. If omitted, inferred from --checkpoint's "
                        "filename (best-effort; a warning is printed if it can't be determined).")
    p.add_argument('--val_fraction', type=float, default=0.1,
                   help="Must match the val_fraction the checkpoint's own training run used "
                        "(params.py's default is 0.1).")
    p.add_argument('--extract_batch_size', type=int, default=256,
                   help="Batch size for feature extraction (not the probe's own training batch size, "
                        "which is fixed at 1024 per the frozen recipe).")
    p.add_argument('--ncc', action='store_true', help="Also compute nearest-class-center forget accuracy")
    p.add_argument('--select_recipe', action='store_true',
                   help="Recipe-selection mode: sweep weight_decay on this (Original) checkpoint's "
                        "validation accuracy and write the result to --recipe_json, instead of "
                        "running a normal probe evaluation.")
    p.add_argument('--gpu_id', type=int, default=0)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[linear_probe] device={device}")

    set_determinism(args.seed)

    model = load_checkpoint_model(args.checkpoint, device)
    assert model.training is False, "Checkpoint must be in eval() mode for extraction."

    expected_in_features = _classifier_in_features(model, args.arch)

    splits = build_splits(args.dataset, args.arch, args.data_dir, args.seed, args.val_fraction)

    # --- Assertion: no index overlap between train/val/test ---------------
    train_idx_set = set(splits.train_indices)
    val_idx_set = set(splits.val_indices)
    assert train_idx_set.isdisjoint(val_idx_set), "Train/val index overlap detected."
    # test is a disjoint on-disk split (torchvision's train=False file) by
    # construction, never drawn from the same index space as train/val.

    train_loader = DataLoader(splits.train, batch_size=args.extract_batch_size, shuffle=False)
    val_loader = DataLoader(splits.val, batch_size=args.extract_batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.extract_batch_size, shuffle=False)

    train_feats, train_labels = get_or_extract_features(
        model, train_loader, device, args.cache_dir, args.checkpoint, 'train')
    val_feats, val_labels = get_or_extract_features(
        model, val_loader, device, args.cache_dir, args.checkpoint, 'val')
    test_feats, test_labels = get_or_extract_features(
        model, test_loader, device, args.cache_dir, args.checkpoint, 'test')

    # --- Assertion: feature dim matches the classifier's in_features -------
    assert train_feats.shape[1] == expected_in_features, (
        f"Extracted feature dim {train_feats.shape[1]} != classifier in_features "
        f"{expected_in_features} for arch={args.arch!r}."
    )

    # --- Assertion: the test split contains forget-class images -----------
    assert (test_labels == args.forget_class).any(), (
        f"No forget-class ({args.forget_class}) images found in the test split - "
        f"check --forget_class and --dataset."
    )

    if args.select_recipe:
        recipe = select_recipe(train_feats, train_labels, val_feats, val_labels,
                               splits.num_classes, device, args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.recipe_json)), exist_ok=True)
        with open(args.recipe_json, 'w') as f:
            json.dump(recipe, f, indent=2)
        print(f"[linear_probe] Recipe written to {args.recipe_json}")
        return recipe

    recipe = load_recipe(args.recipe_json)
    weight_decay = recipe['weight_decay']

    mean, std = compute_standardizer(train_feats)
    train_feats_std = standardize(train_feats, mean, std)
    test_feats_std = standardize(test_feats, mean, std)

    probe = train_probe(train_feats_std, train_labels, splits.num_classes,
                        weight_decay, device, args.seed)

    metrics = evaluate_probe(probe, test_feats_std, test_labels, args.forget_class, device)

    ncc_forget_acc = None
    if args.ncc:
        ncc_forget_acc = ncc_accuracy(train_feats_std, train_labels, test_feats_std, test_labels,
                                      splits.num_classes, args.forget_class)

    method = infer_method_name(args.checkpoint, args.method)
    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'method': method,
        'config': {
            'arch': args.arch,
            'dataset': args.dataset,
            'forget_class': args.forget_class,
            'seed': args.seed,
        },
        'feature_dim': train_feats.shape[1],
        'weight_decay': weight_decay,
        **metrics,
        'ncc_forget_acc': ncc_forget_acc,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[linear_probe] Result written to {args.output_json}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
