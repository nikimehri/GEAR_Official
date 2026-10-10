"""Runs each baseline using the hyperparameters an earlier hpo_search.py run
found for it, evaluated on the full TEST split - validation was already
spent choosing these hyperparameters, so this is the final, honest check
against data no part of the search ever touched. Runs every method across
multiple seeds (fresh train/val split per seed, matching this project's
established "vary --seed for genuinely independent replicates" convention -
see main.py/README.md) and optionally overrides each method's epoch count
to a fixed, comparable training budget regardless of whatever epoch count
the HPO search happened to land on for that method.

Usage:
    python baselines/run_best_hyperparams.py \\
        --data_name cifar10 --model_name resnet18 \\
        --original_model model_checkpoints/resnet18_cifar10_original_checkpoint_0.9411.pth \\
        --retrain_model model_checkpoints/resnet18_cifar10_retrain_checkpoint_0.9417.pth \\
        --forget_class 0 --seeds 42,43,44 \\
        --hpo_csv hpo_cifar10_resnet18_all_baselines_best.csv \\
        --method bad_teacher,cfk,coun,cu,delete,euk,finetune,neggrad,salun,scrub,ssd \\
        --epoch_override 10 \\
        --name final_test_cifar10_resnet18

Design
------
Reuses hpo_search.py's setup()/load_fresh_original_model()/call_unlearn()
directly (same hyperparameter-dict format already stored in --hpo_csv's
'hyperparameters' column, produced by the exact same call_unlearn() this
script calls) rather than re-deriving a mapping onto baseline_main.py's own
per-method CLI flags (--cfk_lr, --scrub_epochs, etc.) - that mapping would
be a second, independent place for a hyperparameter name to drift out of
sync with the search that found it. is_applicable() is reused the same way,
so a method inapplicable to this data_name/model_name (e.g. cheng_unlearn on
cifar10, which has no known class hierarchy - see class_hierarchy.py) is
skipped with a clear message, exactly like during the search itself.

--epoch_override, when given, replaces whatever epoch-controlling
hyperparameter(s) the HPO search picked for each method with this fixed
value (EPOCH_HP_KEYS below maps each method to its own epoch key(s) - they
have different names per method, e.g. finetune's 'epochs' vs. delete's
'unlearn_epoch'). Every other hyperparameter the search found (lr, etc.)
is left untouched. cfk/euk/salun's lr_decay_epochs is NOT itself an
epoch-controlling key - it's a derived LR-schedule milestone pair computed
FROM the epoch count (see hpo_search.py's search spaces) - so overriding
the epoch count without also recomputing lr_decay_epochs would leave a
stale schedule mismatched to the new epoch count (e.g. decaying almost
immediately then sitting flat for the rest of training); LR_DECAY_DERIVED
recomputes it using the exact same formula hpo_search.py's own search
spaces use. ssd has no epoch concept at all (no training loop - two
Fisher-information passes plus a weight-dampening step), so --epoch_override
is a no-op for it.

Every method's result is one row in {name}_full_report.csv via
baseline_main.all_readouts() - test AND validation accuracy, both MIA types,
Retain Adjacent/Remote Accuracy, AIN if --compute_ain, t-SNE if --tsne - all
in one shared file across every method AND every seed (the existing 'Seed'
column distinguishes rows; unlike hpo_search.py, which gives each method its
own file to avoid mixing up per-trial rows, here each (method, seed) pair
runs exactly once, so one shared file is the more useful artifact).

Each (method, seed)'s unlearned model is also saved to
{name}_{method}_seed{seed}.pth, for a later linear_probe.py pass (see
probing/linear_probe.py / probing/aggregate_results.py) - all_readouts()
itself does not save a checkpoint.

A completeness summary prints at the end (and is written to
{name}_completeness.json) - which (method, seed) pairs succeeded vs. were
skipped/failed and why, since a per-(method, seed) try/except means a
failure is silently survivable, not silently invisible.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import hpo_search as hs  # noqa: E402 - reused for setup/load_fresh_original_model/call_unlearn/is_applicable
import baseline_main as bm  # noqa: E402 - reused for all_readouts()


# Maps each method to the hyperparameter key(s) in its HPO-found dict that
# control epoch count - these have different names per method (see
# hpo_search.py's _space_* functions). Empty list = no epoch concept (ssd).
EPOCH_HP_KEYS = {
    'finetune':      ['epochs'],
    'neggrad':       ['epochs'],
    'cfk':           ['cfk_epochs'],
    'euk':           ['euk_epochs'],
    'scrub':         ['sgda_epochs'],
    'delete':        ['unlearn_epoch'],
    'ssd':           [],
    'coun':          ['epochs'],
    'cu':            ['max_epochs'],
    'cheng_unlearn': ['stage1_epochs', 'stage2_epochs'],
    'salun':         ['unlearn_epochs'],
    'bad_teacher':   ['epochs'],
}

# For methods whose hp dict also carries a derived lr_decay_epochs (NOT
# itself suggested by Optuna - computed from the epoch count at search time,
# see hpo_search.py's _space_cfk/_space_euk/_space_salun), recompute it with
# the SAME formula so it stays consistent with an overridden epoch count.
LR_DECAY_DERIVED = {
    'cfk':   lambda epochs: [max(1, epochs // 2), max(2, int(epochs * 0.8))],
    'euk':   lambda epochs: [max(1, epochs // 2), max(2, int(epochs * 0.8))],
    'salun': lambda epochs: [max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))],
}


def apply_epoch_override(method, hp, epoch_override):
    """Returns a COPY of hp with its epoch-controlling key(s) replaced by
    epoch_override (and lr_decay_epochs recomputed to match, where
    applicable) - never mutates the caller's dict. A no-op (returns hp
    unchanged) when epoch_override is None or method has no epoch concept."""
    if epoch_override is None:
        return hp
    keys = EPOCH_HP_KEYS.get(method, [])
    if not keys:
        return hp
    hp = dict(hp)
    for k in keys:
        hp[k] = epoch_override
    if method in LR_DECAY_DERIVED:
        hp['lr_decay_epochs'] = LR_DECAY_DERIVED[method](epoch_override)
    return hp


def _fix_method_column(report_csv, run_name, method):
    """all_readouts()'s `name` argument doubles as both the t-SNE filename/
    print label (which must stay seed-qualified, e.g. 'finetune_seed42', so
    multiple seeds' t-SNE plots don't overwrite each other) AND the CSV's
    'Method' column value (which must NOT be seed-qualified, or grouping by
    method across the 3 seeds - e.g. averaging retain/forget accuracy over
    seeds for 'finetune' - breaks, since 'finetune_seed42'/'finetune_seed43'
    would be three different strings instead of one method's 3 replicates).
    Rewrites just-written row(s) matching run_name back to the plain method
    name, right after the all_readouts() call that wrote them."""
    with open(report_csv, newline='') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    changed = False
    for row in rows:
        if row.get('Method') == run_name:
            row['Method'] = method
            changed = True
    if changed:
        with open(report_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_hyperparams(hpo_csv, method):
    """Reads an hpo_search.py {name}_all_baselines_best.csv and returns the
    winning full hyperparameters dict (not optuna_params - this needs every
    key call_unlearn() consumes, including derived-not-suggested ones like
    cfk/euk/salun's lr_decay_epochs) for `method`, or None if that method has
    no row there."""
    with open(hpo_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['method'] == method:
                return json.loads(row['hyperparameters'])
    return None


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run each baseline once with its HPO-found hyperparameters, evaluated on test")
    p.add_argument('--data_name', type=str, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--original_model', type=str, required=True)
    p.add_argument('--retrain_model', type=str, required=True)
    p.add_argument('--forget_class', type=int, required=True)
    p.add_argument('--seeds', type=str, required=True,
                   help="Comma-separated seeds, e.g. '42,43,44'. Each seed gets its own fresh "
                        "train/val split (torch.utils.data.random_split is reseeded per setup() call) "
                        "AND reseeds numpy/torch globally before that method's own run - matching "
                        "main.py's established 'vary --seed for genuinely independent replicates' "
                        "convention, not just varying the unlearning algorithm's own randomness.")
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--hpo_csv', type=str, required=True,
                   help="Path to an hpo_search.py {name}_all_baselines_best.csv - each requested "
                        "method's winning 'hyperparameters' JSON is read from here.")
    p.add_argument('--method', type=str, required=True,
                   help="Comma-separated list of methods to run (each must have a row in --hpo_csv, "
                        "or be inapplicable to this data_name/model_name - see is_applicable()).")
    p.add_argument('--epoch_override', type=int, default=None,
                   help="Replace each method's HPO-found epoch-controlling hyperparameter(s) with "
                        "this fixed value (see EPOCH_HP_KEYS) - every other found hyperparameter is "
                        "left untouched. No-op for ssd (no epoch concept). Omit to use each method's "
                        "HPO-found epoch count as-is.")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--name', type=str, required=True,
                   help="Output prefix. Shared report: {name}_full_report.csv (one row per "
                        "(method, seed) pair). Checkpoints: {name}_{method}_seed{seed}.pth. "
                        "Completeness summary: {name}_completeness.json.")
    p.add_argument('--compute_ain', action='store_true')
    p.add_argument('--ain_error_range', type=float, default=0.05)
    p.add_argument('--ain_lr', type=float, default=0.1)
    p.add_argument('--ain_max_epochs', type=int, default=10)
    p.add_argument('--ain_eval_interval', type=int, default=50)
    p.add_argument('--tsne', action='store_true',
                   help="Generate a forget-vs-retain t-SNE plot for each method's result.")
    return p


def main():
    args = build_arg_parser().parse_args()
    method_list = [m.strip() for m in args.method.split(',')]
    for m in method_list:
        if m not in hs.METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{m}'. Known methods: {list(hs.METHOD_REGISTRY.keys())}")
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[run_best_hyperparams] device={device}")
    if args.epoch_override is not None:
        print(f"[run_best_hyperparams] epoch_override={args.epoch_override} "
              f"(every other HPO-found hyperparameter is kept as-is)")

    completed = []   # list of (method, seed)
    skipped = []     # list of (method, seed_or_None, reason)

    for seed in seeds:
        print(f"\n{'#'*70}\n  SEED {seed}\n{'#'*70}")
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fresh per-seed: setup()'s train/val split is itself seeded by
        # args.seed, so a different seed must rebuild ctx, not reuse one
        # built for a previous seed - otherwise every seed would train
        # against the exact same split, making them not genuinely
        # independent replicates.
        ctx = hs.setup(args, device)

        for method in method_list:
            if not hs.is_applicable(method, args, ctx['trainset']):
                if seed == seeds[0]:  # only note an inapplicable method once, not once per seed
                    skipped.append((method, None, 'inapplicable to this data_name/model_name'))
                continue

            hp = load_hyperparams(args.hpo_csv, method)
            if hp is None:
                print(f"[run_best_hyperparams] {method}: no row in {args.hpo_csv} - skipping.")
                skipped.append((method, seed, f'no row in {args.hpo_csv}'))
                continue
            hp = apply_epoch_override(method, hp, args.epoch_override)

            run_name = f"{method}_seed{seed}"
            print(f"\n{'='*70}\n  Running {run_name} with hyperparameters: {hp}\n{'='*70}")
            try:
                model = hs.load_fresh_original_model(args, device, ctx['num_classes'])

                start = time.time()
                unlearned = hs.call_unlearn(method, model, args, ctx, hp)
                elapsed = time.time() - start

                bm.all_readouts(unlearned, ctx['test_loader'], ctx['final_forget_loader'],
                                ctx['final_remain_loader'], seed=seed, name=run_name, unlearn_time=elapsed)
                _fix_method_column(f"{args.name}_full_report.csv", run_name, method)

                ckpt_path = f"{args.name}_{run_name}.pth"
                torch.save(unlearned, ckpt_path)
                print(f"[run_best_hyperparams] {run_name}: saved checkpoint to {ckpt_path}")
                completed.append((method, seed))
            except Exception:
                # One (method, seed)'s failure (bad hyperparameters, an OOM,
                # a shape mismatch) shouldn't stop every run after it -
                # print the full traceback for debugging, then move on.
                print(f"[run_best_hyperparams] {run_name}: FAILED, skipping. Traceback:")
                traceback.print_exc()
                skipped.append((method, seed, 'exception during run - see traceback above'))

    expected = len(method_list) * len(seeds)
    print(f"\n{'='*70}\n  COMPLETENESS SUMMARY\n{'='*70}")
    print(f"Completed: {len(completed)}/{expected} expected (method, seed) runs")
    for method, seed in completed:
        print(f"  OK     {method} seed={seed}")
    for method, seed, reason in skipped:
        print(f"  MISSING {method} seed={seed if seed is not None else 'all'}: {reason}")

    summary_path = f"{args.name}_completeness.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'expected_runs': expected,
            'completed': [{'method': m, 'seed': s} for m, s in completed],
            'skipped': [{'method': m, 'seed': s, 'reason': r} for m, s, r in skipped],
        }, f, indent=2)
    print(f"[run_best_hyperparams] Wrote completeness summary to {summary_path}")


if __name__ == '__main__':
    main()
