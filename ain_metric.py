"""Anamnesis Index (AIN) — a corrected implementation, imported by gear.py
and baselines/baseline_main.py alike (one implementation, not two copies).

AIN (Chundawat et al., "Zero-Shot Machine Unlearning,"
https://arxiv.org/abs/2201.05629) measures how much residual information
about the forgotten data survives unlearning, via "relearn time": the number
of mini-batches of fine-tuning on the forgotten data it takes to bring a
model's forget-set accuracy back within error_range of the original
(pre-unlearning) model's forget-set accuracy. A model that has truly
forgotten needs about as long to relearn as a model retrained from scratch
without ever seeing the forgotten data (the gold-standard "retrain" model);
one that hasn't relearns suspiciously fast.

    AIN = relearn_time(unlearned_model) / relearn_time(retrain_model)

AIN close to 1 is good (the unlearned model behaves like the retrain gold
standard); AIN << 1 means the unlearned model relearns much faster than the
gold standard, i.e. it retained meaningfully more information about the
forgotten data than true retraining would have.

This is a corrected reimplementation, not a port, of the reference code at
https://github.com/ayushkumartarun/zero-shot-unlearning/blob/main/metrics.py#L68,
which has several real bugs: relies on a notebook-global `train_ds` instead
of its own `train_data` parameter; `relearn_time`'s `valid_dl` is an
undefined-in-scope global, not the function's own `valid_loader` parameter;
`training_step` is called with the wrong number of arguments; a comment
claims a 4-epoch cap while the code loops `range(10)`; there's no handling
for non-convergence (a model that never reaches target_acc silently returns
the epoch cap as if it had converged) and no zero-division guard on the
final ratio; and its default optimizer/LR (Adam, lr=0.001) doesn't match
what the paper's own text describes (SGD, lr=0.1). This implementation fixes
all of the above; it does not reproduce the paper's plateau-decay LR
schedule, since a fixed learning rate is a reasonable simplification for a
metric utility used over a small, fixed number of relearning steps -
documented here as a deliberate simplification, not an oversight.

Deliberate simplification vs. the reference/paper: the relearning phase
here fine-tunes on whatever relearn_loader the caller passes in - every call
site in this codebase passes train_forget_loader (the forgotten data
specifically), not a reconstructed full original training set as the
reference code technically uses. Re-exposing the model specifically to the
forgotten data is a defensible, common reading of AIN's intent.
"""
import copy
import json
import os

import torch
from torch import nn


def _eval_accuracy(model, loader, device):
    """Plain top-1 accuracy over a full loader pass. Leaves model in eval
    mode on return - fine here, since relearn_time always calls
    model.train() again before its next optimizer step."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1:
                y = y.squeeze()
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def relearn_time(model, relearn_loader, eval_loader, target_acc, device,
                  lr=0.1, max_epochs=10, eval_interval=50):
    """Fine-tunes a deep copy of model on relearn_loader and returns the
    number of mini-batch steps it took for accuracy on eval_loader to reach
    target_acc. Convergence is checked every eval_interval steps (not every
    single one, since evaluation is not free). Returns float('inf') - with a
    printed warning, not a silent fallback - if target_acc is never reached
    within max_epochs. model is not modified in place."""
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}")

    model = copy.deepcopy(model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    step = 0
    for _epoch in range(max_epochs):
        for x, y in relearn_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1:
                y = y.squeeze()

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_interval == 0:
                acc = _eval_accuracy(model, eval_loader, device)
                model.train()
                if acc >= target_acc:
                    return step

    final_acc = _eval_accuracy(model, eval_loader, device)
    if final_acc >= target_acc:
        return step

    print(f"[ain_metric] relearn_time did not converge within {max_epochs} epoch(s) "
          f"({step} steps): final_acc={final_acc:.4f}, target_acc={target_acc:.4f}")
    return float('inf')


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def _save_cache(cache_path, cache):
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def _qualify_cache_key(cache_key, target_acc, lr, max_epochs, eval_interval):
    """Appends the measurement settings to the caller's cache key.

    A cached relearn_time is only reusable by a run that would have measured
    it the same way. The caller's key identifies the *config*
    (dataset/forget_class/seed) but not the *measurement*, so without this a
    run at a different --ain_eval_interval (or lr/max_epochs/error_range,
    which moves target_acc) would silently reuse a gold value computed under
    different settings and report a wrong AIN ratio for every method - with
    no error, since the stale number is a perfectly valid integer.

    Changing any of these now misses the old entry and recomputes, which is
    the safe failure direction: a redundant computation rather than a
    silently incorrect denominator. Old entries are left in place, harmless
    and simply unused.
    """
    if cache_key is None:
        return None
    return (f"{cache_key}|target_acc={target_acc:.6f}|lr={lr}"
            f"|max_epochs={max_epochs}|eval_interval={eval_interval}")


def _gold_relearn_time(retrain_model, relearn_loader, eval_loader, target_acc, device,
                        lr, max_epochs, eval_interval, cache_key, cache_path):
    """Computes (or reuses, via an on-disk cache) the retrain/gold-standard
    model's relearn_time - the expensive half of the AIN ratio, which is the
    same for every method evaluated against a given (dataset, forget_class,
    seed) config *measured the same way* (see _qualify_cache_key).
    Uncached (cache_key=None) recomputes every call."""
    cache_key = _qualify_cache_key(cache_key, target_acc, lr, max_epochs, eval_interval)
    if cache_key is not None:
        cache = _load_cache(cache_path)
        if cache_key in cache:
            print(f"[ain_metric] Using cached gold-standard relearn_time for '{cache_key}': {cache[cache_key]}")
            return cache[cache_key]

    gold_time = relearn_time(retrain_model, relearn_loader, eval_loader, target_acc, device,
                             lr=lr, max_epochs=max_epochs, eval_interval=eval_interval)

    if cache_key is not None:
        cache = _load_cache(cache_path)  # reload in case another run wrote since we last read
        cache[cache_key] = gold_time
        _save_cache(cache_path, cache)

    return gold_time


def compute_ain(unlearned_model, retrain_model, original_model, relearn_loader, eval_loader, device,
                 error_range=0.05, lr=0.1, max_epochs=10, eval_interval=50,
                 cache_key=None, cache_path='ain_gold_cache.json'):
    """Returns AIN = relearn_time(unlearned_model) / relearn_time(retrain_model).

    target_acc is derived from original_model's own accuracy on eval_loader
    (its forget-set accuracy before any unlearning happened), scaled down by
    error_range - the paper's stated 5% margin by default: relearning is
    judged to have "recovered" the forgotten information once accuracy is
    back within that margin of where the original model stood.

    cache_key, when given (e.g. f"{data_name}_{forget_class}_{seed}"),
    caches the retrain model's relearn_time on disk at cache_path so it's
    computed once per config and reused across every method evaluated
    against that config, rather than recomputed on every call.
    """
    original_acc = _eval_accuracy(copy.deepcopy(original_model).to(device), eval_loader, device)
    target_acc = original_acc * (1 - error_range)

    unlearned_time = relearn_time(unlearned_model, relearn_loader, eval_loader, target_acc, device,
                                  lr=lr, max_epochs=max_epochs, eval_interval=eval_interval)
    gold_time = _gold_relearn_time(retrain_model, relearn_loader, eval_loader, target_acc, device,
                                   lr, max_epochs, eval_interval, cache_key, cache_path)

    if gold_time == 0:
        print("[ain_metric] Gold-standard relearn_time is 0 - AIN is undefined; returning float('nan').")
        return float('nan')

    return unlearned_time / gold_time
