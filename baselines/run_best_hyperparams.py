"""Runs each baseline ONCE using the hyperparameters an earlier hpo_search.py
run found for it, evaluated on the full TEST split - validation was already
spent choosing these hyperparameters, so this is the final, honest check
against data no part of the search ever touched.

Usage:
    python baselines/run_best_hyperparams.py \\
        --data_name cifar10 --model_name resnet18 \\
        --original_model model_checkpoints/resnet18_cifar10_original_checkpoint_0.9411.pth \\
        --retrain_model model_checkpoints/resnet18_cifar10_retrain_checkpoint_0.9417.pth \\
        --forget_class 0 --seed 42 \\
        --hpo_csv hpo_cifar10_resnet18_all_baselines_best.csv \\
        --method bad_teacher,cfk,coun,cu,euk,finetune,neggrad,salun,scrub,ssd \\
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

Every method's result is one row in {name}_full_report.csv via
baseline_main.all_readouts() - test AND validation accuracy, both MIA types,
Retain Adjacent/Remote Accuracy, AIN if --compute_ain, t-SNE if --tsne - all
in one shared file (unlike hpo_search.py, which gives each method its own
file to avoid mixing up per-trial rows; here each method runs exactly once,
so one shared file covering every baseline is the more useful artifact).

Each method's unlearned model is also saved to {name}_{method}.pth, for a
later linear_probe.py pass (see probing/linear_probe.py / probing/
aggregate_results.py) - all_readouts() itself does not save a checkpoint.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import hpo_search as hs  # noqa: E402 - reused for setup/load_fresh_original_model/call_unlearn/is_applicable
import baseline_main as bm  # noqa: E402 - reused for all_readouts()


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
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--hpo_csv', type=str, required=True,
                   help="Path to an hpo_search.py {name}_all_baselines_best.csv - each requested "
                        "method's winning 'hyperparameters' JSON is read from here.")
    p.add_argument('--method', type=str, required=True,
                   help="Comma-separated list of methods to run (each must have a row in --hpo_csv, "
                        "or be inapplicable to this data_name/model_name - see is_applicable()).")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--name', type=str, required=True,
                   help="Output prefix. Shared report: {name}_full_report.csv (one row per method). "
                        "Checkpoints: {name}_{method}.pth.")
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

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[run_best_hyperparams] device={device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx = hs.setup(args, device)

    for method in method_list:
        if not hs.is_applicable(method, args, ctx['trainset']):
            continue

        hp = load_hyperparams(args.hpo_csv, method)
        if hp is None:
            print(f"[run_best_hyperparams] {method}: no row in {args.hpo_csv} - skipping.")
            continue

        print(f"\n{'='*70}\n  Running {method} with hyperparameters: {hp}\n{'='*70}")
        try:
            model = hs.load_fresh_original_model(args, device, ctx['num_classes'])

            start = time.time()
            unlearned = hs.call_unlearn(method, model, args, ctx, hp)
            elapsed = time.time() - start

            bm.all_readouts(unlearned, ctx['test_loader'], ctx['final_forget_loader'],
                            ctx['final_remain_loader'], seed=args.seed, name=method, unlearn_time=elapsed)

            ckpt_path = f"{args.name}_{method}.pth"
            torch.save(unlearned, ckpt_path)
            print(f"[run_best_hyperparams] {method}: saved checkpoint to {ckpt_path}")
        except Exception:
            # One method's failure (bad hyperparameters, an OOM, a shape
            # mismatch) shouldn't stop every method after it from running -
            # print the full traceback for debugging, then move on.
            import traceback
            print(f"[run_best_hyperparams] {method}: FAILED, skipping. Traceback:")
            traceback.print_exc()


if __name__ == '__main__':
    main()
