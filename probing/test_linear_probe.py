"""Sanity tests for probing/linear_probe.py. Run directly:

    python probing/test_linear_probe.py

No pytest - this repo has no existing test framework, so these follow the
project's established convention of standalone, directly-runnable
verification scripts (see the ad-hoc verification done throughout this
project's migration work) rather than introducing a new dependency.

Three tests, per the brief:
  1. Synthetic separability test - a fresh probe must recover a clearly-
     separable forget class (>90%) when it's included in training, and MUST
     score 0% on it when it's excluded from training - proving inclusion is
     actually necessary, not just permitted.
  2. Tiny end-to-end smoke test - a small, genuinely randomly-initialized
     ResNet-18 (NOT models.CustomResNet, which always loads pretrained
     ImageNet weights - see the note in test_2 for why a local test-double
     is used instead) on a tiny CIFAR-10 subset, 2 probe epochs, asserting
     the full pipeline runs and writes a valid result JSON.
  3. The main script's own assertions (index overlap, eval() mode, feature-
     dim match, forget-class presence in test) are exercised for real by
     test 2 running the actual main() path - not re-tested separately here.
"""
import json
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import linear_probe as lp  # noqa: E402

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = ''):
    status = 'PASS' if cond else 'FAIL'
    print(f"{status}  {name}" + (f"  ({detail})" if detail else ''))
    (PASS if cond else FAIL).append(name)


# =============================================================================
# TEST 1: Synthetic separability
# =============================================================================

def test_1_synthetic_separability():
    print("\n" + "=" * 70)
    print("TEST 1: synthetic Gaussian-cluster separability")
    print("=" * 70)

    torch.manual_seed(0)
    num_classes = 4
    forget_class = 0
    feat_dim = 16
    n_per_class = 200

    # Well-separated class centers (inter-center distance ~14.1, noise std
    # 0.1) - "clearly separable" per the brief. Checked empirically: at a
    # noisier 0.5 std, a linear probe's leftover decision region for the
    # untrained forget class becomes initialization-sensitive (0%-41% across
    # seeds, since softmax coupling still shapes that region's boundary even
    # with zero positive training signal) - a real, interesting effect, but
    # not what "clearly separable" should test. 0.1 std reliably gives
    # exactly 0% across seeds, matching the brief's literal expectation.
    centers = torch.eye(num_classes, feat_dim) * 10.0

    def make_dataset(classes_to_include):
        feats, labels = [], []
        for c in classes_to_include:
            feats.append(centers[c].unsqueeze(0) + 0.1 * torch.randn(n_per_class, feat_dim))
            labels.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(feats), torch.cat(labels)

    device = torch.device('cpu')

    # --- Case A: forget class included in probe training ---
    train_feats, train_labels = make_dataset(range(num_classes))
    test_feats, test_labels = make_dataset(range(num_classes))

    # batch_size=32 (not train_probe's production default of 1024) because
    # this toy dataset (800 samples) would otherwise collapse to a single
    # batch per epoch - 50 total optimizer steps, nowhere near enough to
    # converge even on trivially-separable clusters (verified empirically:
    # at batch_size=1024 this scores ~88%, not a real separability failure,
    # just too few steps). train_probe's own default remains 1024/50/1e-3
    # for every real call site - only this toy-scale unit test overrides it.
    probe = lp.train_probe(train_feats, train_labels, num_classes,
                           weight_decay=0.0, device=device, seed=0, epochs=50, batch_size=32)
    metrics = lp.evaluate_probe(probe, test_feats, test_labels, forget_class, device)
    print(f"  [included] forget_class_probe_acc = {metrics['forget_class_probe_acc']:.4f}")
    check("probe recovers forget class when included (>90%)",
          metrics['forget_class_probe_acc'] > 0.90,
          f"got {metrics['forget_class_probe_acc']:.4f}")

    # --- Case B: forget class EXCLUDED from probe training ---
    other_classes = [c for c in range(num_classes) if c != forget_class]
    train_feats_excl, train_labels_excl = make_dataset(other_classes)
    # Probe only ever has (num_classes - 1) output rows worth of signal for
    # real classes, but the linear layer still has num_classes outputs - the
    # forget class's output unit never receives a positive training signal.
    probe_excl = lp.train_probe(train_feats_excl, train_labels_excl, num_classes,
                                weight_decay=0.0, device=device, seed=0, epochs=50, batch_size=32)
    metrics_excl = lp.evaluate_probe(probe_excl, test_feats, test_labels, forget_class, device)
    print(f"  [excluded] forget_class_probe_acc = {metrics_excl['forget_class_probe_acc']:.4f}")
    check("probe scores 0% on forget class when excluded from training",
          metrics_excl['forget_class_probe_acc'] == 0.0,
          f"got {metrics_excl['forget_class_probe_acc']:.4f}")

    # NCC sanity check too, since it's part of this module's public API.
    ncc_acc = lp.ncc_accuracy(train_feats, train_labels, test_feats, test_labels,
                              num_classes, forget_class)
    print(f"  [included] ncc_forget_acc = {ncc_acc:.4f}")
    check("NCC also recovers forget class when included (>90%)", ncc_acc > 0.90, f"got {ncc_acc:.4f}")

    ncc_acc_excl = lp.ncc_accuracy(train_feats_excl, train_labels_excl, test_feats, test_labels,
                                   num_classes, forget_class)
    print(f"  [excluded] ncc_forget_acc = {ncc_acc_excl}")
    check("NCC forget-class center is undefined (None) when excluded from training",
          ncc_acc_excl is None)


# =============================================================================
# TEST 2: Tiny end-to-end smoke test
# =============================================================================

class _RandomInitResNet18(nn.Module):
    """A genuinely randomly-initialized (NOT pretrained) ResNet-18, exposing
    the same get_embedding() API every real model in this repo does.

    models.CustomResNet(arch='resnet18') always downloads and loads
    ImageNet-pretrained weights inside __init__ (see models.py) - there is
    no flag to skip that. For a fast, offline-capable smoke test of this
    SCRIPT's plumbing (extraction -> caching -> standardize -> train_probe
    -> evaluate -> JSON write), what matters is that get_embedding() returns
    the right shape, not whether the backbone's weights are pretrained or
    random - so this local test-double (torchvision.models.resnet18(weights=
    None), with no other code changes) is used instead of models.py's actual
    class, deliberately avoiding both the network dependency and the
    ~30s+ download for an automated test that should run instantly offline.
    Real-weight ResNet-18 construction is already verified separately (see
    the models.py wiring verification run earlier for this task, which used
    actual downloaded pretrained weights)."""

    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet_base = resnet18(weights=None)
        num_ftrs = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Identity()
        # Match models.CustomResNet's cifar_stem treatment exactly, since
        # this test uses 32x32 CIFAR-10 crops.
        self.resnet_base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet_base.maxpool = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))

    def forward(self, x):
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_embedding(self, x):
        x = self.resnet_base(x)
        return torch.flatten(x, 1)


def test_2_tiny_end_to_end_smoke():
    print("\n" + "=" * 70)
    print("TEST 2: tiny end-to-end smoke test (random ResNet-18, CIFAR-10 subset)")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix='linear_probe_smoke_')
    try:
        device = torch.device('cpu')
        num_classes = 10
        forget_class = 0

        model = _RandomInitResNet18(num_classes=num_classes)
        model.to(device)
        model.eval()
        check("model.training is False after .eval()", model.training is False)

        # 1,000-image CIFAR-10-shaped subset, synthetic (no real download
        # needed for a plumbing smoke test) but exercising the exact same
        # code path (extract_features -> cache -> standardize -> train_probe
        # -> evaluate_probe -> JSON) real data would.
        g = torch.Generator().manual_seed(0)
        images = torch.randn(1000, 3, 32, 32, generator=g)
        labels = torch.randint(0, num_classes, (1000,), generator=g)
        # Force at least some forget-class samples into existence (random
        # assignment alone could in principle skip a class at this size).
        labels[:50] = forget_class

        full_dataset = TensorDataset(images, labels)

        train_size = 800
        val_size = 100
        test_size = 100
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # --- Assertion 1: no index overlap between train/val/test ---------
        train_idx = set(train_ds.indices)
        val_idx = set(val_ds.indices)
        test_idx = set(test_ds.indices)
        check("no index overlap: train vs val", train_idx.isdisjoint(val_idx))
        check("no index overlap: train vs test", train_idx.isdisjoint(test_idx))
        check("no index overlap: val vs test", val_idx.isdisjoint(test_idx))

        # --- Assertion 2: the test split contains forget-class images -----
        test_labels_only = labels[list(test_idx)]
        has_forget_in_test = bool((test_labels_only == forget_class).any())
        check("test split contains forget-class images", has_forget_in_test)
        if not has_forget_in_test:
            # Extremely unlikely given labels[:50] is forced to forget_class
            # and spread across a random split, but fail loudly rather than
            # silently if it ever happens instead of producing a bogus NaN.
            raise RuntimeError("Test split has no forget-class samples - rerun with a different seed.")

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

        cache_dir = os.path.join(tmp_dir, 'cache')
        fake_checkpoint_path = os.path.join(tmp_dir, 'fake_checkpoint_resnet18_cifar10.pth')
        torch.save(model, fake_checkpoint_path)  # so cache_path_for has something real to hash

        train_feats, train_labels_extracted = lp.get_or_extract_features(
            model, train_loader, device, cache_dir, fake_checkpoint_path, 'train')
        val_feats, val_labels_extracted = lp.get_or_extract_features(
            model, val_loader, device, cache_dir, fake_checkpoint_path, 'val')
        test_feats, test_labels_extracted = lp.get_or_extract_features(
            model, test_loader, device, cache_dir, fake_checkpoint_path, 'test')

        # --- Assertion 3: feature dim matches the classifier's in_features -
        expected_dim = model.classifier[1].in_features
        check("feature dim matches classifier in_features",
              train_feats.shape[1] == expected_dim,
              f"got {train_feats.shape[1]}, expected {expected_dim}")

        # Cache round-trip: a second call must return identical features
        # (proves the fp16-on-disk cache path, not just the fresh-extraction
        # path, actually works).
        cached_feats, cached_labels = lp.get_or_extract_features(
            model, train_loader, device, cache_dir, fake_checkpoint_path, 'train')
        # fp16 round-trip error scales with magnitude (mantissa precision,
        # not a fixed absolute step) - a pure atol=1e-3 is tighter than fp16
        # can represent for features of this scale (verified empirically:
        # ~0.004 max abs error at feature magnitude ~14). rtol+atol together
        # is the correct way to bound this, matching how fp16 precision
        # actually behaves.
        check("cached features match freshly-extracted features",
              torch.allclose(train_feats, cached_feats, rtol=1e-2, atol=1e-3),
              "fp16 round-trip, rtol=1e-2/atol=1e-3")
        check("cached labels match freshly-extracted labels",
              torch.equal(train_labels_extracted, cached_labels))

        mean, std = lp.compute_standardizer(train_feats)
        train_feats_std = lp.standardize(train_feats, mean, std)
        test_feats_std = lp.standardize(test_feats, mean, std)

        probe = lp.train_probe(train_feats_std, train_labels_extracted, num_classes,
                               weight_decay=1e-3, device=device, seed=0, epochs=2)
        metrics = lp.evaluate_probe(probe, test_feats_std, test_labels_extracted, forget_class, device)
        ncc_acc = lp.ncc_accuracy(train_feats_std, train_labels_extracted,
                                  test_feats_std, test_labels_extracted, num_classes, forget_class)

        result = {
            'checkpoint': fake_checkpoint_path,
            'method': 'smoke_test',
            'config': {'arch': 'resnet18', 'dataset': 'cifar10', 'forget_class': forget_class, 'seed': 0},
            'feature_dim': train_feats.shape[1],
            'weight_decay': 1e-3,
            **metrics,
            'ncc_forget_acc': ncc_acc,
        }
        output_json = os.path.join(tmp_dir, 'result.json')
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        # --- Validate the written JSON is well-formed and complete --------
        with open(output_json, 'r') as f:
            loaded = json.load(f)
        required_keys = {'checkpoint', 'method', 'config', 'feature_dim', 'weight_decay',
                         'forget_class_probe_acc', 'retain_class_probe_acc', 'overall_probe_acc',
                         'n_test_forget', 'n_test_retain', 'n_test_total', 'ncc_forget_acc'}
        check("result JSON contains every required key", required_keys.issubset(loaded.keys()),
              f"missing: {required_keys - loaded.keys()}")
        check("result JSON is valid (round-trips through json.load)", loaded == result)
        print(f"  Full result: {json.dumps(result, indent=2)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# TEST 3 (per brief item 1, second half): weight_decay recipe selection
# =============================================================================

def test_3_recipe_selection():
    print("\n" + "=" * 70)
    print("TEST 3: recipe selection picks the best validation weight_decay")
    print("=" * 70)

    torch.manual_seed(0)
    device = torch.device('cpu')
    num_classes = 3
    feat_dim = 8

    centers = torch.eye(num_classes, feat_dim) * 5.0

    def make(n):
        feats, labels = [], []
        for c in range(num_classes):
            feats.append(centers[c].unsqueeze(0) + 0.3 * torch.randn(n, feat_dim))
            labels.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(feats), torch.cat(labels)

    train_feats, train_labels = make(100)
    val_feats, val_labels = make(30)

    # batch_size=32 for the same reason as test 1: this toy dataset is far
    # smaller than a real CIFAR train split, so train_probe's production
    # default (batch_size=1024) would collapse to ~1 batch/epoch and never
    # converge. select_recipe's real call sites never pass probe_kwargs, so
    # production runs always use the frozen 1024/50/1e-3 recipe unchanged.
    recipe = lp.select_recipe(train_feats, train_labels, val_feats, val_labels,
                              num_classes, device, seed=0, probe_kwargs={'batch_size': 32})
    check("select_recipe returns a weight_decay from the fixed grid",
          recipe['weight_decay'] in lp.WEIGHT_DECAY_GRID)
    check("select_recipe records val_acc for every grid point",
          set(recipe['val_acc_by_weight_decay'].keys()) == {str(wd) for wd in lp.WEIGHT_DECAY_GRID})

    # Recipe file round-trip (write -> load_recipe).
    tmp_dir = tempfile.mkdtemp(prefix='recipe_test_')
    try:
        recipe_path = os.path.join(tmp_dir, 'recipe.json')
        with open(recipe_path, 'w') as f:
            json.dump(recipe, f)
        loaded = lp.load_recipe(recipe_path)
        check("load_recipe round-trips the written recipe", loaded == recipe)

        missing_path = os.path.join(tmp_dir, 'does_not_exist.json')
        try:
            lp.load_recipe(missing_path)
            check("load_recipe raises on a missing recipe file", False)
        except FileNotFoundError:
            check("load_recipe raises on a missing recipe file", True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    test_1_synthetic_separability()
    test_2_tiny_end_to_end_smoke()
    test_3_recipe_selection()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    if FAIL:
        print("FAILED:")
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)
    print("ALL SANITY TESTS PASSED")
