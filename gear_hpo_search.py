"""TPE Optuna hyperparameter search for GEAR itself (not the baselines -
see baselines/hpo_search.py for those). Searches beta (remain_reg_param),
gamma_rep, retain_forget_weight, forget_forget_weight, feature_align_weight,
and poison_epoch, evaluated on the held-out VALIDATION split - test stays
untouched throughout the search, same discipline as baselines/hpo_search.py.

Usage:
    python gear_hpo_search.py \\
        --data_name cifar10 --model_name resnet18 \\
        --original_model <original_model.pth> --retrain_model <retrain_model.pth> \\
        --forget_class 0 --seed 42 \\
        --forget_forget_mode disperse \\
        --n_trials 20 \\
        --name gear_hpo_cifar10_resnet18_disperse

Design
------
forget_forget_mode is a FIXED setting for the whole search (default
'disperse', matching the explicit request this script was built for), not
itself a search dimension - pass --forget_forget_mode attract to search
that regime instead. use_entanglement_weighting/feature_contrastive are
fixed True (required for any of the searched weights to have an effect at
all - see gear.py's `if feature_contrastive:` gate) and target_layer is
fixed per --target_layer (default 'layer4', the established ResNet
convention; AllCNN configs should pass --target_layer 9).

Selection rule and TPESampler settings are identical to baselines/
hpo_search.py's, for consistency: among trials with val_forget_acc <=
forget_acc_threshold, pick the highest val_retain_acc (adaptive fallback to
the lowest observed val_forget_acc if none clear the threshold);
TPESampler(multivariate=True, group=True) for joint/relative sampling.

gear.gear() already does its own full per-trial CSV logging when given
results_csv (every hyperparameter plus every result metric in one row - see
gear.py's row dict) - every trial's call shares the same results_csv path,
so {name}_full_report.csv accumulates one row per trial with zero extra
logging code needed here, mirroring how baselines/hpo_search.py reuses
baseline_main.all_readouts()'s own CSV writing instead of duplicating it.

Cost warning: a single GEAR epoch is noticeably heavier than a baseline
unlearning epoch (the additional contrastive/entanglement machinery adds
real overhead - a 2-epoch GEAR run on the cheapest available config,
AllCNN/CIFAR-10, took ~940-990s/~16min on CPU during this project's own
local verification). poison_epoch is itself searched here (2-10), so a
20-trial budget on CIFAR-10/ResNet18 is a genuinely large amount of compute
- a GPU is not optional the way it arguably was for some of the cheaper
baselines, it's required to finish this in a reasonable time.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

import gear  # noqa: E402
from make_dataloaders import get_dataset, dataloader_engine  # noqa: E402
from utils import set_num_classes  # noqa: E402
from trainer import load_checkpoint, eval as trainer_eval  # noqa: E402

PENALTY_WEIGHT = 10.0


def setup(args, device):
    """Mirrors main.py's own single-experiment setup block exactly (same
    val_fraction/seed convention) - duplicated here rather than refactored
    out of main.py, same reasoning baselines/hpo_search.py's setup() already
    documented: lower risk than touching complex, already-verified script
    code."""
    trainset_full, testset, dataset = get_dataset(args.data_name, args.dataset_dir, model_name=args.model_name)

    val_size = int(len(trainset_full) * args.val_fraction)
    train_size = len(trainset_full) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=split_generator
    )

    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
    test_loader_real = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    num_classes, idx_to_class = set_num_classes(args, dataset)
    total_forget_class = sum(1 for _, target in trainset if target == args.forget_class)

    train_forget_loader, train_remain_loader, val_forget_loader, val_remain_loader, \
        test_forget_loader, test_remain_loader, _, _, _, _, _, _, \
        train_dict, val_dict, test_dict = dataloader_engine(
            args, trainset, valset, testset, None, num_forget=total_forget_class,
            oculoplastics=args.oculoplastics, selective_unlearning=False,
        )

    return dict(
        trainset=trainset, valset=valset, testset=testset, dataset=dataset,
        num_classes=num_classes, idx_to_class=idx_to_class,
        train_forget_loader=train_forget_loader, train_remain_loader=train_remain_loader,
        val_loader=val_loader, test_loader_real=test_loader_real,
        test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader,
        train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
    )


def make_objective(args, ctx, device):
    results_csv = f"{args.name}_full_report.csv"

    def objective(trial):
        hp = dict(
            remain_reg_param=trial.suggest_float('remain_reg_param', 0.1, 5.0),
            gamma_rep=trial.suggest_float('gamma_rep', 0.1, 5.0),
            retain_forget_weight=trial.suggest_float('retain_forget_weight', 0.1, 10.0, log=True),
            forget_forget_weight=trial.suggest_float('forget_forget_weight', 0.1, 10.0, log=True),
            feature_align_weight=trial.suggest_float('feature_align_weight', 0.0, 2.0),
            poison_epoch=trial.suggest_int('poison_epoch', 2, 10),
            # Targets Retain Adjacent Accuracy directly (see
            # migration_notes/experiments_plan.md section 17): makes the
            # entanglement weighting symmetric, protecting retain samples
            # closest to the forget class - the ones most likely to absorb
            # collateral damage from forget_forget_weight/retain_forget_weight.
            # Previously fixed at its gear.gear() default (False) here,
            # meaning this search never actually explored it.
            retain_entanglement_protection=trial.suggest_categorical(
                'retain_entanglement_protection', [True, False]),
            retain_protection_strength=trial.suggest_float('retain_protection_strength', 0.5, 5.0, log=True),
        )

        ori_model = load_checkpoint(args.original_model, 'original_model', device=device)
        retrain_model = load_checkpoint(args.retrain_model, 'retrain_model', device=device) \
            if args.retrain_model else None

        start = time.time()
        unlearn_model, forget_acc, remain_acc, gear_time, test_acc, mia_mean, \
            retain_adjacent_acc, retain_remote_acc, ain_score, full_metrics = gear.gear(
                ori_model, ctx['train_forget_loader'], ctx['trainset'], ctx['valset'], ctx['val_loader'], device,
                test_metadata=ctx['val_dict'], train_metadata=ctx['train_dict'],
                output_name=f"{args.name}_trial{trial.number}",
                retrain_model=retrain_model, train_remain_loader=ctx['train_remain_loader'],
                remain_reg_param=hp['remain_reg_param'],
                feature_contrastive=True,
                feature_align_weight=hp['feature_align_weight'],
                retain_forget_weight=hp['retain_forget_weight'],
                forget_forget_weight=hp['forget_forget_weight'],
                forget_forget_mode=args.forget_forget_mode,
                retain_entanglement_protection=hp['retain_entanglement_protection'],
                retain_protection_strength=hp['retain_protection_strength'],
                gamma_rep=hp['gamma_rep'],
                target_layer=args.target_layer,
                results_csv=results_csv,
                use_entanglement_weighting=True,
                num_classes=ctx['num_classes'],
                data_name=args.data_name,
                poison_epoch=hp['poison_epoch'],
                forget_class=args.forget_class,
                seed=args.seed,
                generate_tsne=False,
            )
        elapsed = time.time() - start

        # val_forget_acc/val_retain_acc: forget_acc/remain_acc ARE validation
        # accuracy here, since valset/val_loader were passed into gear()'s
        # dv/test_loader parameter slots - the same convention main.py's own
        # --specific_settings branch uses.
        val_forget_acc = forget_acc.item() if isinstance(forget_acc, torch.Tensor) else forget_acc
        val_retain_acc = remain_acc.item() if isinstance(remain_acc, torch.Tensor) else remain_acc

        # Test-set numbers are recorded for reporting ONLY, never used for
        # selection (the search never looks at test) - a cheap extra eval()
        # pass on the already-trained unlearn_model, not a second gear() run.
        _, test_forget_acc_t = trainer_eval(model=unlearn_model, data_loader=ctx['test_forget_loader'], device=device)
        _, test_remain_acc_t = trainer_eval(model=unlearn_model, data_loader=ctx['test_remain_loader'], device=device)
        test_forget_acc = test_forget_acc_t.item() if isinstance(test_forget_acc_t, torch.Tensor) else test_forget_acc_t
        test_retain_acc = test_remain_acc_t.item() if isinstance(test_remain_acc_t, torch.Tensor) else test_remain_acc_t

        trial.set_user_attr('val_forget_acc', val_forget_acc)
        trial.set_user_attr('val_retain_acc', val_retain_acc)
        trial.set_user_attr('test_forget_acc', test_forget_acc)
        trial.set_user_attr('test_retain_acc', test_retain_acc)
        trial.set_user_attr('hyperparams', hp)
        trial.set_user_attr('optuna_params', dict(trial.params))
        trial.set_user_attr('run_time', elapsed)

        penalty = PENALTY_WEIGHT * max(0.0, val_forget_acc - args.forget_acc_threshold)
        return val_retain_acc - penalty

    return objective


def select_best_trial(study, forget_acc_threshold, retain_floor_frac=0.5):
    """Identical selection rule to baselines/hpo_search.py's, for
    consistency: among trials with val_forget_acc <= forget_acc_threshold,
    pick the highest val_retain_acc; adaptive fallback to the lowest
    observed val_forget_acc if none clear the threshold.

    Same guard as baselines/hpo_search.py's select_best_trial against
    crowning a collapsed trial winner: a model whose retain accuracy has
    crashed toward chance level trivially gets near-zero forget accuracy
    too (failing every class, not selectively forgetting one) - large
    forget_forget_weight/retain_forget_weight values can do this to GEAR
    just as badly as the equivalent baselines, so this needs the same
    self-relative retain floor before the stated rule is applied."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None, forget_acc_threshold

    best_retain_seen = max(t.user_attrs['val_retain_acc'] for t in completed)
    retain_floor = retain_floor_frac * best_retain_seen
    healthy = [t for t in completed if t.user_attrs['val_retain_acc'] >= retain_floor]
    if not healthy:
        print(f"[gear_hpo_search] WARNING: every completed trial has val_retain_acc below "
              f"{retain_floor_frac} x the best observed ({best_retain_seen:.4f}) - this run "
              f"looks collapsed in every trial, not just the one that would otherwise be "
              f"selected. Falling back to the full (unfiltered) trial pool, but the result "
              f"should not be trusted without investigating why.")
        healthy = completed

    candidates = [t for t in healthy if t.user_attrs['val_forget_acc'] <= forget_acc_threshold]
    effective_threshold = forget_acc_threshold
    if not candidates:
        effective_threshold = min(t.user_attrs['val_forget_acc'] for t in healthy)
        candidates = [t for t in healthy if t.user_attrs['val_forget_acc'] == effective_threshold]
    best = max(candidates, key=lambda t: t.user_attrs['val_retain_acc'])
    return best, effective_threshold


def main():
    p = argparse.ArgumentParser(description="TPE Optuna hyperparameter search for GEAR")
    p.add_argument('--data_name', type=str, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--original_model', type=str, required=True)
    p.add_argument('--retrain_model', type=str, default=None,
                   help="Optional - only used for GEAR's own retrain-baseline comparison print/log, "
                        "not required for the search to run.")
    p.add_argument('--forget_class', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--dataset_dir', type=str, default='./data')
    p.add_argument('--oculoplastics', action='store_true', default=False)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--forget_acc_threshold', type=float, default=0.02,
                   help="x in the selection rule: among trials with val_forget_acc <= x, pick the "
                        "highest val_retain_acc. Falls back to the lowest observed val_forget_acc "
                        "if no trial clears this threshold.")
    p.add_argument('--retain_floor_frac', type=float, default=0.5,
                   help="Trials with val_retain_acc below this fraction of the study's own best "
                        "observed val_retain_acc are excluded before the selection rule runs - "
                        "guards against picking a collapsed trial. See select_best_trial()'s docstring.")
    p.add_argument('--forget_forget_mode', type=str, default='disperse', choices=['attract', 'disperse'],
                   help="FIXED for the whole search (not itself swept) - default 'disperse'.")
    p.add_argument('--target_layer', type=str, default='layer4',
                   help="FIXED for the whole search. 'layer4' for resnet/resnet50/resnet18, '9' for "
                        "AllCNN, a 0-indexed transformer block number for ViT/DistilBERT.")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--name', type=str, required=True,
                   help="Output prefix. Per-trial reports: {name}_full_report.csv (written by "
                        "gear.gear() itself). Summary: {name}_best.csv. Overall best hyperparameters: "
                        "{name}_best_hyperparameters.json.")
    args = p.parse_args()
    # dataloader_engine's branch check (see make_dataloaders.py) - this
    # script only supports class-based forgetting, not metadata-based.
    args.custom_unlearn = False

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[gear_hpo_search] device={device}")
    print(f"[gear_hpo_search] forget_forget_mode={args.forget_forget_mode} (fixed for this whole search)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx = setup(args, device)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=args.name)
    study.optimize(make_objective(args, ctx, device), n_trials=args.n_trials)

    best_trial, effective_threshold = select_best_trial(study, args.forget_acc_threshold, args.retain_floor_frac)
    if best_trial is None:
        print("[gear_hpo_search] No trials completed.")
        return

    if effective_threshold != args.forget_acc_threshold:
        print(f"[gear_hpo_search] No trial reached forget_acc <= {args.forget_acc_threshold} - "
              f"fell back to the lowest observed forget_acc ({effective_threshold:.4f}) as the threshold.")

    rows = []
    for t in sorted((tr for tr in study.trials if tr.state == optuna.trial.TrialState.COMPLETE),
                    key=lambda tr: tr.user_attrs['val_retain_acc'], reverse=True):
        rows.append({
            'Trial': t.number,
            'Selected': (t.number == best_trial.number),
            'Val Forget Acc': t.user_attrs['val_forget_acc'],
            'Val Retain Acc': t.user_attrs['val_retain_acc'],
            'Test Forget Acc': t.user_attrs['test_forget_acc'],
            'Test Retain Acc': t.user_attrs['test_retain_acc'],
            'Run Time': t.user_attrs['run_time'],
            'Hyperparameters': json.dumps(t.user_attrs['hyperparams']),
        })
    summary_path = f"{args.name}_best.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[gear_hpo_search] Wrote {summary_path} ({len(rows)} trials, winner=trial {best_trial.number})")

    best_path = f"{args.name}_best_hyperparameters.json"
    with open(best_path, 'w') as f:
        json.dump({
            'forget_forget_mode': args.forget_forget_mode,
            'target_layer': args.target_layer,
            'hyperparameters': best_trial.user_attrs['hyperparams'],
            'val_forget_acc': best_trial.user_attrs['val_forget_acc'],
            'val_retain_acc': best_trial.user_attrs['val_retain_acc'],
            'test_forget_acc': best_trial.user_attrs['test_forget_acc'],
            'test_retain_acc': best_trial.user_attrs['test_retain_acc'],
        }, f, indent=2)
    print(f"[gear_hpo_search] Wrote {best_path}")
    print(json.dumps(best_trial.user_attrs['hyperparams'], indent=2))


if __name__ == '__main__':
    main()
