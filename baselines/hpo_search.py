"""Hyperparameter search for every baseline, using Optuna's TPE sampler,
evaluated on the held-out VALIDATION split (never the test split - test
stays untouched throughout the entire search, for an honest final check
later).

Usage (one baseline):
    python baselines/hpo_search.py \\
        --data_name cifar10 --model_name allcnn \\
        --original_model <original_model.pth> --retrain_model <retrain_model.pth> \\
        --forget_class 0 --seed 42 \\
        --method delete \\
        --n_trials 35 \\
        --forget_acc_threshold 0.02 \\
        --name hpo_cifar10_allcnn

Usage (every baseline this script knows how to search):
    ... --method all ...

Design
------
Selection rule (applied EXACTLY as specified, as a post-hoc filter over the
real recorded validation metrics - not approximated by whatever Optuna's
internal objective happens to optimize): among trials with
val_forget_acc <= forget_acc_threshold, pick the one with the highest
val_retain_acc. If NO trial clears the threshold, the threshold falls back
to the lowest val_forget_acc actually observed across all trials (so the
candidate pool is never empty), and the highest-val_retain_acc trial among
THOSE ties is picked instead.

Optuna objective (what TPE actually searches toward - a separate thing
from the selection rule above): a single scalar,
val_retain_acc - PENALTY_WEIGHT * max(0, val_forget_acc - forget_acc_threshold).
This gives TPE a smooth, continuous signal to climb (unlike the selection
rule's hard threshold, which would make every failing trial look equally
bad to the sampler, wasting its sample budget). The final winner is still
chosen by the EXACT selection rule above, applied after the study
completes - the penalized objective only guides the search, it never
overrides the stated rule.

TPESampler is configured with multivariate=True, group=True. This is what
makes Optuna actually call infer_relative_search_space internally and
model hyperparameters jointly (relative/correlated sampling) rather than
independently one-at-a-time - group=True is specifically needed here
because different baselines have different hyperparameter sets with no
fixed shared shape, and group-relative sampling handles that correctly.
There's no supported way to call infer_relative_search_space directly from
user code - TPESampler already calls it internally once multivariate=True
is set, which is the actual, correct way to use this feature.

Every trial still gets a full baseline_main.all_readouts() report - a row
in {name}_{method}_full_report.csv (test AND validation metrics, both MIA
types, AIN if --compute_ain, Retain Adjacent/Remote Accuracy) plus its own
t-SNE plot, exactly like every other baseline run - the search doesn't
skip any of that, it just additionally uses two of those columns
(val_forget_error/val_retain_error) to drive the search and the selection
rule.

Output per baseline (in addition to the per-trial full_report.csv/t-SNE):
  {name}_{method}_best.csv   - one row per trial (hyperparameters + metrics
                               + whether it was selected), sorted best-first
Output once, after every requested baseline has been searched:
  {name}_all_baselines_best.csv - one row per baseline, its chosen
                                  hyperparameters (JSON-encoded, since each
                                  baseline has a different hyperparameter
                                  set) and the metrics that won.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import class_hierarchy  # noqa: E402

import baseline_main as bm  # noqa: E402 - reused for all_readouts() and its globals
from baseline_utils import (  # noqa: E402
    get_dataset, get_dataloader, set_num_classes, load_model, load_model_state, get_forget_loader,
)
from finetune import finetune
from neggrad import negative_grad
from euk import cfk_unlearn, euk_unlearn, CFK_EUK_SUPPORTED_MODELS
from scrub import scrub_unlearn
from delete import delete_unlearn
from ssd import ssd_unlearn
from coun import get_coun_datasets, coun_unlearn, COUN_VALID_PAIRINGS
from cu import cu_unlearn
from cheng_unlearn import cheng_unlearn
from salun import salun_unlearn
from bad_teacher import bad_teacher_unlearn

PENALTY_WEIGHT = 10.0  # see module docstring - scales how hard the search
                       # penalizes exceeding forget_acc_threshold


# =============================================================================
# SETUP (self-contained, mirrors baseline_main.py's single-experiment setup -
# not reused directly since that setup is inline script code there, not a
# callable function; duplicating this ~30-line block was judged lower-risk
# than refactoring baseline_main.py's existing, already-verified dispatch).
# =============================================================================

def setup(args, device):
    """Builds every loader/model-loading piece a search needs, and sets the
    module-level globals baseline_main.all_readouts() relies on. Returns a
    dict of everything the objective functions need directly."""
    trainset_full, testset, dataset = get_dataset(args.data_name, './data', model_name=args.model_name)

    val_size = int(len(trainset_full) * args.val_fraction)
    train_size = len(trainset_full) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=split_generator
    )

    batch_size = args.batch_size
    train_loader, test_loader = get_dataloader(trainset, testset, batch_size, device=device)
    num_classes, idx_to_class = set_num_classes(args.data_name, dataset)
    total_forget_class = sum(1 for _, target in dataset if target == args.forget_class)

    from baseline_utils import dataloader_engine
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict = (
        dataloader_engine(batch_size, trainset, testset, None,
                          num_forget=total_forget_class, forget_class=args.forget_class,
                          oculoplastics=False, custom_unlearn=False, selective_unlearning=False)
    )

    final_forget_loader, final_remain_loader = get_forget_loader(testset, args.forget_class)
    val_forget_loader, val_remain_loader = get_forget_loader(valset, args.forget_class)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    full_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    orig_model_path = args.original_model
    retrain_model_path = args.retrain_model

    # --- Wire baseline_main.all_readouts()'s module-level globals ----------
    bm.device = device
    bm.num_classes = num_classes
    bm.idx_to_class = idx_to_class
    bm.data_name = args.data_name
    bm.dataset = dataset
    bm.forget_class = args.forget_class
    bm.model_type = args.model_name
    bm.train_forget_loader = train_forget_loader
    bm.orig_model_path = orig_model_path
    bm.retrain_model_path = retrain_model_path
    bm.val_loader = val_loader
    bm.val_forget_loader = val_forget_loader
    bm.val_remain_loader = val_remain_loader
    bm.args = args  # all_readouts reads args.compute_ain / args.tsne / args.ain_* / args.name

    return dict(
        trainset=trainset, testset=testset, dataset=dataset, valset=valset,
        num_classes=num_classes, batch_size=batch_size,
        train_loader=train_loader, test_loader=test_loader, val_loader=val_loader,
        train_forget_loader=train_forget_loader, train_remain_loader=train_remain_loader,
        final_forget_loader=final_forget_loader, final_remain_loader=final_remain_loader,
        val_forget_loader=val_forget_loader, val_remain_loader=val_remain_loader,
        full_train_loader=full_train_loader,
        train_remain_index=train_remain_index,
    )


def load_fresh_original_model(args, device, num_classes):
    """Every trial must start from the SAME pristine original checkpoint -
    never a model another trial already mutated."""
    model = load_model(args.model_name, num_classes=num_classes, data_name=args.data_name).to(device)
    model = load_model_state(model, args.original_model)
    return model


# =============================================================================
# SEARCH SPACES - one suggest_fn(trial) -> kwargs dict per baseline.
# =============================================================================

def _space_finetune(trial, ctx):
    return dict(lr=trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                epochs=trial.suggest_int('epochs', 1, 15))


def _space_neggrad(trial, ctx):
    one_minus_alpha = trial.suggest_float('one_minus_alpha', 1e-5, 0.5, log=True)
    return dict(alpha=1.0 - one_minus_alpha,
                lr=trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                epochs=trial.suggest_int('epochs', 1, 15))


def _space_cfk(trial, ctx):
    epochs = trial.suggest_int('cfk_epochs', 1, 20)
    return dict(cfk_lr=trial.suggest_float('cfk_lr', 1e-4, 1e-1, log=True),
                cfk_epochs=epochs,
                lr_decay_epochs=(max(1, epochs // 2), max(2, int(epochs * 0.8))))


def _space_euk(trial, ctx):
    epochs = trial.suggest_int('euk_epochs', 1, 20)
    return dict(euk_lr=trial.suggest_float('euk_lr', 1e-4, 1e-1, log=True),
                euk_epochs=epochs,
                lr_decay_epochs=(max(1, epochs // 2), max(2, int(epochs * 0.8))))


def _space_scrub(trial, ctx):
    return dict(sgda_epochs=trial.suggest_int('sgda_epochs', 1, 10),
                remain_reg=trial.suggest_float('remain_reg', 0.5, 5.0),
                beta_ce=trial.suggest_float('beta_ce', 0.0, 1.0))


def _space_delete(trial, ctx):
    return dict(unlearn_epoch=trial.suggest_int('unlearn_epoch', 1, 30),
                unlearn_rate=trial.suggest_float('unlearn_rate', 1e-5, 1e-2, log=True),
                disable_bn=trial.suggest_categorical('disable_bn', [True, False]))


def _space_ssd(trial, ctx):
    return dict(dampening_constant=trial.suggest_float('dampening_constant', 0.1, 10.0, log=True),
                selection_weighting=trial.suggest_float('selection_weighting', 1.0, 50.0, log=True))


def _space_coun(trial, ctx):
    return dict(lambda_scale=trial.suggest_float('lambda_scale', 0.1, 6.0),
                temp=trial.suggest_float('temp', 0.05, 0.3),
                epochs=trial.suggest_int('epochs', 1, 5),
                lr=trial.suggest_float('lr', 1e-3, 1e-1, log=True))


def _space_cu(trial, ctx):
    return dict(lambda_ul=trial.suggest_float('lambda_ul', 0.1, 5.0),
                lambda_ce=trial.suggest_float('lambda_ce', 0.1, 5.0),
                temperature=trial.suggest_float('temperature', 0.01, 1.0, log=True),
                omega=trial.suggest_int('omega', 1, 8),
                lr=trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                max_epochs=trial.suggest_int('max_epochs', 1, 15))


def _space_cheng_unlearn(trial, ctx):
    return dict(stage1_epochs=trial.suggest_int('stage1_epochs', 1, 5),
                stage1_lr=trial.suggest_float('stage1_lr', 1e-7, 1e-4, log=True),
                mu=trial.suggest_float('mu', 1.0, 50.0, log=True),
                gamma=trial.suggest_float('gamma', 0.1, 5.0),
                c=trial.suggest_float('c', 1.0, 50.0),
                stage2_epochs=trial.suggest_int('stage2_epochs', 1, 15),
                stage2_lr=trial.suggest_float('stage2_lr', 1e-6, 1e-3, log=True),
                momentum=trial.suggest_float('momentum', 0.5, 0.99),
                alpha=trial.suggest_float('alpha', 0.1, 0.9))


def _space_salun(trial, ctx):
    epochs = trial.suggest_int('unlearn_epochs', 1, 15)
    return dict(mask_ratio=trial.suggest_float('mask_ratio', 0.1, 0.9),
                unlearn_epochs=epochs,
                lr=trial.suggest_float('lr', 1e-3, 1e-1, log=True),
                momentum=trial.suggest_float('momentum', 0.5, 0.99),
                weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                lr_decay_epochs=(max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))))


def _space_bad_teacher(trial, ctx):
    return dict(epochs=trial.suggest_int('epochs', 1, 10),
                lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                temperature=trial.suggest_float('temperature', 0.5, 5.0, log=True))


# =============================================================================
# METHOD REGISTRY: method_name -> space_fn. Applicability checks (which
# methods don't apply to a given data_name/model_name) live in
# is_applicable() below, since cheng_unlearn's check needs trainset.
# =============================================================================

METHOD_REGISTRY = {
    'finetune':     _space_finetune,
    'neggrad':      _space_neggrad,
    'cfk':          _space_cfk,
    'euk':          _space_euk,
    'scrub':        _space_scrub,
    'delete':       _space_delete,
    'ssd':          _space_ssd,
    'coun':         _space_coun,
    'cu':           _space_cu,
    'cheng_unlearn': _space_cheng_unlearn,
    'salun':        _space_salun,
    'bad_teacher':  _space_bad_teacher,
}


def call_unlearn(method, model, args, ctx, hp):
    """Dispatches to the right unlearn function with the trial's suggested
    hyperparameters (hp) plus whatever fixed context each method needs."""
    device = bm.device
    if method == 'finetune':
        finetune(model, ctx['train_remain_loader'], **hp)
        return model
    if method == 'neggrad':
        negative_grad(model, ctx['train_remain_loader'], ctx['train_forget_loader'], **hp)
        return model
    if method == 'cfk':
        return cfk_unlearn(model, ctx['train_remain_loader'], args.model_name, **hp)
    if method == 'euk':
        return euk_unlearn(model, ctx['train_remain_loader'], args.model_name, **hp)
    if method == 'scrub':
        model_s, model_s_final = scrub_unlearn(
            model, model, ctx['train_remain_loader'], ctx['train_forget_loader'],
            args.model_name, args.data_name, **hp,
        )
        return model_s_final
    if method == 'delete':
        return delete_unlearn(model, ctx['train_forget_loader'], device, **hp)
    if method == 'ssd':
        return ssd_unlearn(model, ctx['train_forget_loader'], ctx['full_train_loader'], device,
                           model_name=args.model_name, **hp)
    if method == 'coun':
        _, _, trainset_coun_raw = get_coun_datasets(args.data_name, args.model_name, './data')
        train_remain_loader_raw = DataLoader(trainset_coun_raw, batch_size=ctx['batch_size'],
                                             sampler=SubsetRandomSampler(ctx['train_remain_index']))
        return coun_unlearn(model, args.model_name, args.data_name, train_remain_loader_raw, device, **hp)
    if method == 'cu':
        return cu_unlearn(model, ctx['train_forget_loader'], ctx['train_remain_loader'], device,
                          ctx['num_classes'], eval_forget_loader=ctx['val_forget_loader'], **hp)
    if method == 'cheng_unlearn':
        adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(
            args.data_name, args.forget_class, ctx['trainset'])
        if adjacent_indices is None or remote_indices is None:
            raise RuntimeError(f"cheng_unlearn not applicable to data_name='{args.data_name}' "
                               f"(no known class hierarchy)")
        adjacent_loader = DataLoader(ctx['trainset'], batch_size=ctx['batch_size'],
                                     sampler=SubsetRandomSampler(adjacent_indices))
        remote_loader = DataLoader(ctx['trainset'], batch_size=ctx['batch_size'],
                                   sampler=SubsetRandomSampler(remote_indices))
        return cheng_unlearn(model, ctx['train_forget_loader'], adjacent_loader, remote_loader, device, **hp)
    if method == 'salun':
        return salun_unlearn(model, ctx['train_forget_loader'], ctx['train_remain_loader'], device,
                             ctx['num_classes'], **hp)
    if method == 'bad_teacher':
        return bad_teacher_unlearn(model, ctx['train_forget_loader'], ctx['train_remain_loader'],
                                   args.model_name, ctx['num_classes'], args.data_name, device, **hp)
    raise ValueError(f"Unknown method '{method}'")


# =============================================================================
# OBJECTIVE + SELECTION RULE
# =============================================================================

def make_objective(method, args, ctx):
    space_fn = METHOD_REGISTRY[method]

    def objective(trial):
        hp = space_fn(trial, ctx)
        model = load_fresh_original_model(args, bm.device, ctx['num_classes'])

        start = time.time()
        unlearned = call_unlearn(method, model, args, ctx, hp)
        elapsed = time.time() - start

        name = f"{method}_trial{trial.number}"
        result = bm.all_readouts(unlearned, ctx['test_loader'], ctx['final_forget_loader'],
                                 ctx['final_remain_loader'], seed=args.seed, name=name, unlearn_time=elapsed)

        # NOTE: all_readouts' dict keys are named *_error for historical
        # reasons but store ACCURACY (not error rate) - val_forget_error IS
        # val forget accuracy. See baseline_main.py's all_readouts().
        val_forget_acc = result['val_forget_error']
        val_retain_acc = result['val_retain_error']

        trial.set_user_attr('val_forget_acc', val_forget_acc)
        trial.set_user_attr('val_retain_acc', val_retain_acc)
        trial.set_user_attr('test_forget_acc', result['forget_error'])
        trial.set_user_attr('test_retain_acc', result['retain_error'])
        trial.set_user_attr('hyperparams', hp)
        trial.set_user_attr('run_time', elapsed)

        penalty = PENALTY_WEIGHT * max(0.0, val_forget_acc - args.forget_acc_threshold)
        return val_retain_acc - penalty

    return objective


def select_best_trial(study, forget_acc_threshold):
    """Applies the EXACT selection rule the user specified, as a post-hoc
    filter over the real recorded validation metrics - independent of
    whatever the penalized objective internally optimized toward."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None, forget_acc_threshold
    candidates = [t for t in completed if t.user_attrs['val_forget_acc'] <= forget_acc_threshold]
    effective_threshold = forget_acc_threshold
    if not candidates:
        effective_threshold = min(t.user_attrs['val_forget_acc'] for t in completed)
        candidates = [t for t in completed if t.user_attrs['val_forget_acc'] == effective_threshold]
    best = max(candidates, key=lambda t: t.user_attrs['val_retain_acc'])
    return best, effective_threshold


# =============================================================================
# PER-BASELINE SEARCH + SUMMARY CSV
# =============================================================================

def run_study(method, args, ctx):
    print(f"\n{'='*70}\n  Searching: {method}  ({args.n_trials} trials)\n{'='*70}")

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                study_name=f"{args.name}_{method}")

    # Give all_readouts a method-specific CSV prefix so this method's 35
    # trial rows accumulate into their own full_report.csv, not mixed with
    # other baselines'.
    args.name_for_reports = f"{args.name}_{method}"
    _orig_name = bm.args.name
    bm.args.name = args.name_for_reports

    try:
        study.optimize(make_objective(method, args, ctx), n_trials=args.n_trials)
    finally:
        bm.args.name = _orig_name

    best_trial, effective_threshold = select_best_trial(study, args.forget_acc_threshold)
    if best_trial is None:
        print(f"[hpo_search] {method}: no trials completed - skipping summary.")
        return None

    if effective_threshold != args.forget_acc_threshold:
        print(f"[hpo_search] {method}: no trial reached forget_acc <= {args.forget_acc_threshold} - "
              f"fell back to the lowest observed forget_acc ({effective_threshold:.4f}) as the threshold.")

    # --- Per-baseline summary CSV: one row per trial, best-first ----------
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
    import csv as csv_mod
    summary_path = f"{args.name}_{method}_best.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hpo_search] {method}: wrote {summary_path} ({len(rows)} trials, "
          f"winner=trial {best_trial.number})")

    return {
        'method': method,
        'n_trials_completed': len(rows),
        'forget_acc_threshold_used': effective_threshold,
        'val_forget_acc': best_trial.user_attrs['val_forget_acc'],
        'val_retain_acc': best_trial.user_attrs['val_retain_acc'],
        'test_forget_acc': best_trial.user_attrs['test_forget_acc'],
        'test_retain_acc': best_trial.user_attrs['test_retain_acc'],
        'hyperparameters': json.dumps(best_trial.user_attrs['hyperparams']),
    }


def is_applicable(method, args, trainset):
    if method in ('cfk', 'euk') and args.model_name not in CFK_EUK_SUPPORTED_MODELS:
        print(f"[hpo_search] {method}: not applicable to model_name='{args.model_name}'. Skipping.")
        return False
    if method == 'coun' and args.model_name not in COUN_VALID_PAIRINGS.get(args.data_name, []):
        print(f"[hpo_search] coun: not applicable to data_name='{args.data_name}'/"
              f"model_name='{args.model_name}'. Skipping.")
        return False
    if method == 'cheng_unlearn':
        adjacent, remote = class_hierarchy.get_adjacent_remote_split(args.data_name, args.forget_class, trainset)
        if adjacent is None or remote is None:
            print(f"[hpo_search] cheng_unlearn: no class hierarchy for data_name='{args.data_name}'. Skipping.")
            return False
    return True


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description="TPE Optuna hyperparameter search for GEAR baselines")
    p.add_argument('--data_name', type=str, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--original_model', type=str, required=True)
    p.add_argument('--retrain_model', type=str, required=True)
    p.add_argument('--forget_class', type=int, required=True)
    p.add_argument('--seed', type=int, required=True,
                   help="Used for the TPE sampler, the train/val split, and all_readouts() - "
                        "the SAME seed is used for every baseline searched in one invocation.")
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--method', type=str, default='all',
                   help="Comma-separated list of methods to search, or 'all' for every method "
                        f"in the registry ({', '.join(METHOD_REGISTRY.keys())}).")
    p.add_argument('--n_trials', type=int, default=35,
                   help="Same trial budget used for every baseline searched in this invocation.")
    p.add_argument('--forget_acc_threshold', type=float, default=0.02,
                   help="x in the selection rule: among trials with val_forget_acc <= x, pick the "
                        "highest val_retain_acc. Falls back to the lowest observed val_forget_acc "
                        "if no trial clears this threshold.")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--name', type=str, required=True,
                   help="Output prefix. Per-trial reports: {name}_{method}_full_report.csv and "
                        "{name}_{method}_<trial>_tsne.png. Per-method summary: "
                        "{name}_{method}_best.csv. Overall summary: {name}_all_baselines_best.csv.")
    p.add_argument('--compute_ain', action='store_true',
                   help="Also compute AIN per trial (expensive - see main.py's --compute_ain). "
                        "Off by default since it isn't part of the selection rule.")
    p.add_argument('--ain_error_range', type=float, default=0.05)
    p.add_argument('--ain_lr', type=float, default=0.1)
    p.add_argument('--ain_max_epochs', type=int, default=10)
    p.add_argument('--ain_eval_interval', type=int, default=50)
    p.add_argument('--tsne', action='store_true', default=True,
                   help="Generate a t-SNE plot per trial. On by default for this script "
                        "specifically (every other baseline entry point defaults this off) - a "
                        "35-trial search already pays for every trial's forward passes, so the "
                        "marginal cost of one extra embedding pass per trial is small by "
                        "comparison, and the brief explicitly asks for a t-SNE per trial.")
    p.add_argument('--save_checkpoints', action='store_true',
                   help="Save every trial's unlearned model checkpoint. Off by default - a full "
                        "sweep (n_trials x n_methods) would otherwise write a very large number "
                        "of checkpoint files.")
    return p


def main():
    args = build_arg_parser().parse_args()
    args.method_list = (list(METHOD_REGISTRY.keys()) if args.method == 'all'
                        else [m.strip() for m in args.method.split(',')])
    for m in args.method_list:
        if m not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{m}'. Known methods: {list(METHOD_REGISTRY.keys())}")

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[hpo_search] device={device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx = setup(args, device)

    overall_rows = []
    for method in args.method_list:
        if not is_applicable(method, args, ctx['trainset']):
            continue
        summary = run_study(method, args, ctx)
        if summary is not None:
            overall_rows.append(summary)

    if overall_rows:
        import csv as csv_mod
        overall_path = f"{args.name}_all_baselines_best.csv"
        with open(overall_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overall_rows)
        print(f"\n[hpo_search] Wrote {overall_path} ({len(overall_rows)} baselines)")


if __name__ == '__main__':
    main()
