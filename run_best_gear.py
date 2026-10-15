"""Runs GEAR once per seed with the hyperparameters an earlier
gear_hpo_search.py run found, evaluated on the full TEST split - validation
was already spent choosing these hyperparameters, so this is the final,
honest check against data no part of the search ever touched. Mirrors
main.py's --run_sota branch exactly (same gear.gear() call, same dt/dv/
test_loader slots), and mirrors baselines/run_best_hyperparams.py's design
on the baselines side: reuses gear_hpo_search.py's setup() rather than
duplicating it, and reads hyperparameters straight out of an existing
gear_hpo_search.py {name}_best.csv's Selected row rather than re-deriving a
mapping onto main.py's own CLI flags (--remain_reg, --gamma_rep, etc.).

Usage:
    python run_best_gear.py \\
        --data_name cifar10 --model_name resnet18 \\
        --original_model model_checkpoints/resnet18_cifar10_original_model_25_final_model_<ACC>.pth \\
        --retrain_model model_checkpoints/resnet18_cifar10_retrain_model_class_0_25_final_model_<ACC>.pth \\
        --forget_class 0 --seeds 42,43,44 \\
        --gear_best_csv gear_hpo_cifar10_resnet18_best.csv \\
        --forget_forget_mode disperse --target_layer layer4 \\
        --name final_test_cifar10_resnet18_gear

Design
------
forget_forget_mode/target_layer are NOT read from --gear_best_csv - they're
fixed settings for the whole search (not themselves swept - see
gear_hpo_search.py's module docstring), so they never appear in the CSV's
'Hyperparameters' JSON column. Pass the same values here that the search in
--gear_best_csv used, or this run won't actually match the config that
hyperparameter search was run against.

Every seed's run writes one row to {name}_full_report.csv via gear.gear()'s
own results_csv logging (gear.py's native schema - output_name/beta/
gamma_rep/forget_acc/remain_acc/etc., distinct from baseline_main.
all_readouts()'s schema, same as every other GEAR run in this project) and
saves its own checkpoint to {name}_gear_seed{seed}.pth (gear.gear() does
this itself - see its own torch.save(unlearn_model, output_name + '.pth')
call - no extra save needed here, unlike baselines/run_best_hyperparams.py,
which calls all_readouts() rather than gear() and so must save explicitly).
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

sys.path.insert(0, os.path.dirname(__file__))

import gear  # noqa: E402
import gear_hpo_search as ghs  # noqa: E402 - reused for setup()
from trainer import load_checkpoint  # noqa: E402


def load_best_hyperparams(gear_best_csv):
    """Reads an existing gear_hpo_search.py {name}_best.csv and returns the
    Selected row's full hyperparameters dict (JSON-decoded from its
    'Hyperparameters' column)."""
    with open(gear_best_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['Selected'] in ('True', 'true', '1'):
                return json.loads(row['Hyperparameters'])
    raise ValueError(f"No row with Selected=True found in {gear_best_csv}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run GEAR once per seed with its HPO-found hyperparameters, evaluated on test")
    p.add_argument('--data_name', type=str, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--original_model', type=str, required=True)
    p.add_argument('--retrain_model', type=str, default=None,
                   help="Optional - only used for GEAR's own retrain-baseline comparison "
                        "print/log, same as gear_hpo_search.py's --retrain_model.")
    p.add_argument('--forget_class', type=int, required=True)
    p.add_argument('--seeds', type=str, required=True,
                   help="Comma-separated seeds, e.g. '42,43,44' - matches "
                        "baselines/run_best_hyperparams.py's 'vary --seed for genuinely "
                        "independent replicates' convention.")
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--dataset_dir', type=str, default='./data')
    p.add_argument('--oculoplastics', action='store_true', default=False)
    p.add_argument('--gear_best_csv', type=str, required=True,
                   help="Path to a gear_hpo_search.py {name}_best.csv - the Selected row's "
                        "hyperparameters are read from here.")
    p.add_argument('--forget_forget_mode', type=str, default='disperse', choices=['attract', 'disperse'],
                   help="Must match whatever --forget_forget_mode the search in --gear_best_csv "
                        "used (fixed for the whole search, not itself swept - see module docstring).")
    p.add_argument('--target_layer', type=str, default='layer4',
                   help="Must match whatever --target_layer the search used - same reasoning as "
                        "--forget_forget_mode above. 'layer4' for resnet/resnet18/resnet50, '9' for "
                        "AllCNN.")
    p.add_argument('--compute_ain', action='store_true')
    p.add_argument('--ain_error_range', type=float, default=0.05)
    p.add_argument('--ain_lr', type=float, default=0.1)
    p.add_argument('--ain_max_epochs', type=int, default=10)
    p.add_argument('--ain_eval_interval', type=int, default=50)
    p.add_argument('--tsne', action='store_true')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--name', type=str, required=True,
                   help="Output prefix. Shared report: {name}_full_report.csv (one row per seed). "
                        "Checkpoints: {name}_gear_seed{seed}.pth.")
    return p


def main():
    args = build_arg_parser().parse_args()
    args.custom_unlearn = False  # dataloader_engine's branch check - class-based forgetting only
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    hp = load_best_hyperparams(args.gear_best_csv)
    print(f"[run_best_gear] Loaded hyperparameters from {args.gear_best_csv}: {hp}")

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[run_best_gear] device={device}")

    completed, skipped = [], []
    for seed in seeds:
        print(f"\n{'#'*70}\n  SEED {seed}\n{'#'*70}")
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        run_name = f"{args.name}_gear_seed{seed}"
        try:
            # Fresh per-seed, same reasoning as baselines/run_best_hyperparams.py:
            # a different seed must rebuild ctx, not reuse one built for a
            # previous seed, or every seed would train against the exact
            # same train/val/test split.
            ctx = ghs.setup(args, device)

            ori_model = load_checkpoint(args.original_model, 'original_model', device=device)
            retrain_model = load_checkpoint(args.retrain_model, 'retrain_model', device=device) \
                if args.retrain_model else None

            start = time.time()
            unlearn_model, forget_acc, remain_acc, gear_time, test_acc, mia_score, \
                retain_adjacent_acc, retain_remote_acc, ain_score, full_metrics = gear.gear(
                    ori_model, ctx['train_forget_loader'], ctx['trainset'], ctx['testset'],
                    ctx['test_loader_real'], device,
                    test_metadata=ctx['test_dict'], train_metadata=ctx['train_dict'], output_name=run_name,
                    retrain_model=retrain_model, train_remain_loader=ctx['train_remain_loader'],
                    remain_reg_param=hp['remain_reg_param'],
                    feature_contrastive=True,
                    feature_align_weight=hp['feature_align_weight'],
                    retain_forget_weight=hp['retain_forget_weight'],
                    forget_forget_weight=hp['forget_forget_weight'],
                    forget_forget_mode=args.forget_forget_mode,
                    gamma_rep=hp['gamma_rep'],
                    target_layer=args.target_layer,
                    results_csv=f"{args.name}_full_report.csv",
                    use_entanglement_weighting=True,
                    num_classes=ctx['num_classes'],
                    data_name=args.data_name,
                    poison_epoch=hp['poison_epoch'],
                    forget_class=args.forget_class,
                    compute_ain=args.compute_ain,
                    ain_error_range=args.ain_error_range,
                    ain_lr=args.ain_lr,
                    ain_max_epochs=args.ain_max_epochs,
                    ain_eval_interval=args.ain_eval_interval,
                    seed=seed,
                    generate_tsne=args.tsne,
                )
            elapsed = time.time() - start
            forget_acc = forget_acc.item() if isinstance(forget_acc, torch.Tensor) else forget_acc
            remain_acc = remain_acc.item() if isinstance(remain_acc, torch.Tensor) else remain_acc
            print(f"[run_best_gear] {run_name}: done in {elapsed:.1f}s "
                  f"(test forget_acc={forget_acc:.4f}, test remain_acc={remain_acc:.4f}) - "
                  f"checkpoint saved to {run_name}.pth")
            completed.append(seed)
        except Exception:
            # One seed's failure shouldn't stop every seed after it - print
            # the full traceback for debugging, then move on, same
            # survivable-failure convention as run_best_hyperparams.py.
            print(f"[run_best_gear] {run_name}: FAILED, skipping. Traceback:")
            traceback.print_exc()
            skipped.append(seed)

    print(f"\n{'='*70}\n  COMPLETENESS SUMMARY\n{'='*70}")
    print(f"Completed: {len(completed)}/{len(seeds)} expected seeds")
    for s in completed:
        print(f"  OK      seed={s}")
    for s in skipped:
        print(f"  MISSING seed={s}: exception during run - see traceback above")


if __name__ == '__main__':
    main()
