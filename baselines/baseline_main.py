import argparse
import time
import numpy as np
import torch
from scrub import scrub_unlearn
from euk import cfk_unlearn, euk_unlearn, CFK_EUK_SUPPORTED_MODELS
from neggrad import *
from finetune import finetune
from delete import delete_unlearn
from ssd import ssd_unlearn
from coun import get_coun_datasets, coun_unlearn, COUN_VALID_PAIRINGS
from cu import cu_unlearn
from cheng_unlearn import cheng_unlearn
from salun import salun_unlearn
from bad_teacher import bad_teacher_unlearn
from baseline_utils import *
from models import *
import class_hierarchy
import ain_metric
import embeddings
from gear import compute_loss_threshold_mia, log_results_to_csv

from path_dicts import model_paths, selective_forget_models,chen_paths,ravi_paths,med_unlearn_paths, MODEL_CHECKPOINT_ROOT
from tqdm import tqdm


def test(model, loader, idx_to_class, num_classes, device):
    """Per-class accuracy over a full loader pass, returned as
    {class_name: accuracy}."""
    model.eval()
    correct = [0] * num_classes
    cnt = [0] * num_classes
    class_accuracies = {}

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(loader, leave=False)):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for i in range(target.size(0)):
                label = target[i].item()
                if pred[i].item() == label:
                    correct[label] += 1
                cnt[label] += 1

    for i in range(num_classes):
        accuracy = 0. if cnt[i] == 0 else correct[i] / cnt[i]
        class_name = idx_to_class[i]
        class_accuracies[class_name] = accuracy
    return class_accuracies



_ain_model_cache = {}


def save_baseline_checkpoint(model, method_label, model_type, data_name, forget_class, seed, checkpoint_dir):
    """Saves a baseline's final unlearned model as a full pickled model
    object (not a state_dict) - the same convention trainer.py/gear.py
    already use for original/retrain/GEAR checkpoints, so any downstream
    consumer (e.g. a linear-probe script) handles every checkpoint source
    identically via torch.load + the usual nn.DataParallel-unwrap check.

    Filename encodes every field a consumer needs to identify the run
    without parsing free-text: {model_type}_{data_name}_{method_label}_
    forget{forget_class}_seed{seed}.pth. Returns the path written.

    Unlike trainer.py/gear.py's saves, these are NOT nn.DataParallel-wrapped
    (baseline_utils.load_model never wraps its models) - downstream code
    must handle both wrapped and unwrapped checkpoints, which is already
    the established pattern everywhere else in this codebase
    (hasattr(model, 'module') / isinstance(model, nn.DataParallel))."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"{model_type}_{data_name}_{method_label}_forget{forget_class}_seed{seed}.pth"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(model, path)
    print(f"[save_checkpoints] Saved {method_label} checkpoint to {path}")
    return path


def _load_ain_reference_model(path, model_type, num_classes, data_name, device):
    """Loads a checkpoint for AIN's original_model/retrain_model references,
    memoized by path so the same checkpoint isn't reloaded on every single
    all_readouts() call within one experiment (there are up to ~8 per
    experiment: finetune/neggrad/cfk/euk/scrub x2/chen/ravi/retrain).
    Returns None if path is falsy."""
    if not path:
        return None
    if path not in _ain_model_cache:
        ref_model = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
        ref_model = load_model_state(ref_model, path)
        ref_model.eval()
        _ain_model_cache[path] = ref_model
    return _ain_model_cache[path]


def all_readouts(model, test_loader, final_forget_loader, final_remain_loader, seed=2022, name='method',
                  unlearn_time=None):
    """Standard "report card" for any unlearned model: overall test/forget/
    remain accuracy, per-class accuracy, Retain Adjacent/Remote Accuracy, AIN
    (opt-in via --compute_ain), and BOTH membership-inference-attack types
    (confidence-vector-based and loss-threshold-based - see gear.py's
    compute_mia/compute_loss_threshold_mia docstrings for what distinguishes
    them). Also appends one row to {args.name}_full_report.csv covering
    every metric here plus unlearn_time (the caller's responsibility to
    measure and pass in - this function only records it) and, when --tsne
    is set, a forget-vs-retain t-SNE plot path. Called once per baseline
    method after it's finished running. Like the rest of this module, relies
    on device/num_classes/idx_to_class/data_name/dataset/forget_class/args/
    train_forget_loader/orig_model_path/retrain_model_path already being set
    as module-level globals by the caller (single-experiment or sweep-mode
    block) before this is invoked."""
    _, test_acc = eval(model=model, data_loader=test_loader, device=device, name='test set all class')
    _, forget_acc = eval(model=model, data_loader=final_forget_loader, device=device, name='test set forget class')
    _, remain_acc = eval(model=model, data_loader=final_remain_loader, device=device, name='test set remain class')

    # Validation-set accuracy (val_loader/val_forget_loader/val_remain_loader
    # are module-level globals set by the single-experiment setup block,
    # same convention as dataset/device/etc. above) - the held-out split
    # every baseline's own training data excludes, see that block's
    # val_fraction split.
    _, val_acc = eval(model=model, data_loader=val_loader, device=device, name='val set all class')
    _, val_forget_acc = eval(model=model, data_loader=val_forget_loader, device=device, name='val set forget class')
    _, val_remain_acc = eval(model=model, data_loader=val_remain_loader, device=device, name='val set remain class')

    per_class_accs = test(model, test_loader, idx_to_class, num_classes, device)

    # Retain Adjacent/Remote Accuracy (class-taxonomy-based - see
    # class_hierarchy.py; 'N/A' for any data_name without a known hierarchy).
    adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(
        data_name, forget_class, dataset)
    retain_adjacent_acc, retain_remote_acc = class_hierarchy.compute_split_accuracy(
        model, dataset, adjacent_indices, remote_indices, device)

    # AIN (Anamnesis Index) - opt-in and requires both the original and
    # retrain checkpoints (retrain is the gold-standard denominator of the
    # AIN ratio); 'N/A' otherwise, same convention as the metric above.
    ain_score = 'N/A'
    if args.compute_ain:
        original_model = _load_ain_reference_model(orig_model_path, model_type, num_classes, data_name, device)
        retrain_model_for_ain = _load_ain_reference_model(retrain_model_path, model_type, num_classes, data_name, device)
        if original_model is not None and retrain_model_for_ain is not None:
            cache_key = f"{data_name}_{forget_class}_{seed}"
            ain_score = ain_metric.compute_ain(
                model, retrain_model_for_ain, original_model, train_forget_loader, final_forget_loader, device,
                error_range=args.ain_error_range, lr=args.ain_lr, max_epochs=args.ain_max_epochs,
                eval_interval=args.ain_eval_interval, cache_key=cache_key, cache_path='ain_gold_cache.json',
            )
        else:
            print(f"[AIN] Missing orig_model_path/retrain_model_path for '{name}' - reporting N/A.")

    MIA = membership_inference_attack(model, test_loader, final_forget_loader, device, seed=seed, name=name)
    lt_mia_result = compute_loss_threshold_mia(model, train_forget_loader, final_forget_loader, device)

    # t-SNE (opt-in via --tsne) - same forget-vs-retain-only, class-count-
    # agnostic plot gear.py's GEAR runs can now also generate (see
    # embeddings.plot_forget_retain_tsne's docstring). Uses the test-set
    # forget/remain loaders, matching gear.py's own choice for its
    # equivalent plot exactly.
    tsne_path = 'N/A'
    if getattr(args, 'tsne', False):
        tsne_path = embeddings.plot_forget_retain_tsne(
            model, final_forget_loader, final_remain_loader, device,
            output_path=f"{args.name}_{name}_tsne.png",
            title=f"{name} - Forget vs. Retain (t-SNE)",
        )
        print(f"[tsne] Saved to {tsne_path}")

    run_time = unlearn_time if unlearn_time is not None else 'N/A'

    print(f"{name} -> Full test Acc: {test_acc:.5f} Forget Acc: {forget_acc:.5f} Remain Acc: {remain_acc:.5f} "
          f"MIA: {np.mean(MIA):.2f}±{np.std(MIA):0.2f} "
          f"MIA(loss-thr): acc={lt_mia_result['mia_acc']:.4f} auc={lt_mia_result['mia_auc']:.4f} "
          f"RunTime: {run_time}")

    # One row per method, covering every requested metric (run time, forget/
    # retain/test accuracy, both MIA types, AIN, Retain Adjacent/Remote
    # Accuracy, t-SNE path) in a schema matching main.py's GEAR CSV column
    # names, so GEAR and baseline results can be directly compared.
    log_results_to_csv(f"{args.name}_full_report.csv", {
        'Dataset': data_name,
        'Model': model_type,
        'Method': name,
        'Seed': seed,
        'Run Time': run_time,
        'Forget Acc': float(forget_acc),
        'Remain Acc': float(remain_acc),
        'Test Acc': float(test_acc),
        'Val Acc': float(val_acc),
        'Val Forget Acc': float(val_forget_acc),
        'Val Remain Acc': float(val_remain_acc),
        'MIA Confidence Mean': float(np.mean(MIA)),
        'MIA Confidence Std': float(np.std(MIA)),
        'MIA Loss-Threshold AUC': lt_mia_result['mia_auc'],
        'MIA Loss-Threshold Acc': lt_mia_result['mia_acc'],
        'AIN': ain_score,
        'Retain Adjacent Acc': retain_adjacent_acc,
        'Retain Remote Acc': retain_remote_acc,
        'TSNE Plot Path': tsne_path,
    })

    return dict(
        test_error=float(test_acc),
        forget_error=float(forget_acc),
        retain_error=float(remain_acc),
        val_error=float(val_acc),
        val_forget_error=float(val_forget_acc),
        val_retain_error=float(val_remain_acc),
        retain_adjacent_acc=retain_adjacent_acc,
        retain_remote_acc=retain_remote_acc,
        AIN=ain_score,
        MIA_mean=float(np.mean(MIA)),
        MIA_std=float(np.std(MIA)),
        MIA_loss_threshold_auc=lt_mia_result['mia_auc'],
        MIA_loss_threshold_acc=lt_mia_result['mia_acc'],
        run_time=run_time,
        tsne_path=tsne_path,
        per_class=per_class_accs
    )


def load_retrain_forget_acc(retrain_model_path, model_type, num_classes, data_name, forget_loader, device):
    """Loads the retrain (gold-standard) checkpoint, if a path was given, and
    returns its accuracy on forget_loader as a plain float. Used to give
    scrub_unlearn a real target for its checkpoint-selection heuristic
    (select the epoch whose forget accuracy is closest to what a model
    retrained from scratch would have achieved). Returns None if no
    retrain_model_path is available, so scrub_unlearn falls back to its
    default (last-epoch) behavior."""
    if not retrain_model_path:
        return None
    retrain_model = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
    retrain_model = load_model_state(retrain_model, retrain_model_path)
    _, retrain_forget_acc = eval(model=retrain_model, data_loader=forget_loader, device=device)
    return retrain_forget_acc.item() if isinstance(retrain_forget_acc, torch.Tensor) else float(retrain_forget_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Baseline Unlearning")
    # Experiment configuration
    parser.add_argument('--data_name', type=str, default=None,
                        help='Dataset name (e.g. cifar10, cifar100, fashionmnist)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model type (e.g. resnet50, AllCNN)')
    parser.add_argument('--original_model', type=str, default=None,
                        help='Path to original trained model checkpoint')
    parser.add_argument('--retrain_model', type=str, default=None,
                        help='Path to retrained (gold-standard) model checkpoint')
    parser.add_argument('--forget_class', type=int, default=0,
                        help='Class index to forget')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loaders')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Seeds NumPy/PyTorch/CUDA globally and is passed to every '
                             "all_readouts() call (MIA's cross-validation split, AIN's cache key). "
                             'Vary this across runs for genuinely independent stochastic replicates.')
    parser.add_argument('--name', type=str, default='baseline',
                        help='Experiment name prefix for output files')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of the training set held out as a validation split '
                             '(single-experiment mode only), seeded by --seed - matches main.py/'
                             "GEAR's own val_fraction convention and default. Every baseline's "
                             'own training data (train_remain_loader etc.) is built from the '
                             'remaining 1-val_fraction portion, not the full training set.')
    parser.add_argument('--method', type=str, default='scrub',
                        help='Comma-separated list of methods to run, e.g. scrub or finetune,scrub')
    # SCRUB + CL+ES arguments
    parser.add_argument('--feature_contrastive', action='store_true',
                        help='Enable feature-space contrastive losses in SCRUB')
    parser.add_argument('--use_entanglement_weighting', action='store_true',
                        help='Enable entanglement-score weighting for rf_loss and targeted CE in SCRUB')
    parser.add_argument('--retain_forget_weight', type=float, default=2.0,
                        help='Weight for retain-forget cosine similarity loss')
    parser.add_argument('--forget_forget_weight', type=float, default=3.0,
                        help='Weight for forget-forget cosine similarity loss')
    parser.add_argument('--feature_align_weight', type=float, default=0.0,
                        help='Weight for retain-to-original feature alignment loss')
    parser.add_argument('--gamma_rep', type=float, default=1.0,
                        help='Weight on the combined contrastive/representation loss')
    parser.add_argument('--remain_reg', type=float, default=3.5,
                        help='Weight on the retain loss in SCRUB')
    parser.add_argument('--centroid_refresh_interval', type=int, default=None,
                        help='Steps between retain-centroid refreshes (default: one epoch)')
    parser.add_argument('--beta_ce', type=float, default=0.0,
                        help='Weight for targeted CE loss on the retain batch')
    parser.add_argument('--target_layer', type=str, default='layer4',
                        help='Layer for contrastive loss. Use "layer4" for ResNet, "9" for AllCNN, '
                             '"all" for multi-layer ResNet')
    parser.add_argument('--scrub_epochs', type=int, default=5,
                        help='Number of SCRUB training epochs (sgda_epochs). Default: 5.')

    # --- AIN (Anamnesis Index) arguments - mirrors params.py's flags ---
    parser.add_argument('--compute_ain', action='store_true',
                        help='Compute the Anamnesis Index (AIN) metric for every baseline. Off by '
                             'default - unlike the other metrics, this involves actually retraining '
                             'a copy of the model, not just an extra evaluation pass.')
    parser.add_argument('--ain_error_range', type=float, default=0.05,
                        help='AIN relearning target: fraction below the original model\'s '
                             'forget-set accuracy considered "recovered" (paper default: 0.05).')
    parser.add_argument('--ain_lr', type=float, default=0.1,
                        help='Learning rate for AIN\'s relearning-phase SGD optimizer.')
    parser.add_argument('--ain_max_epochs', type=int, default=10,
                        help='Max epochs of relearning before AIN reports non-convergence (inf).')
    parser.add_argument('--ain_eval_interval', type=int, default=50,
                        help='Mini-batch steps between AIN relearning-accuracy checks.')

    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save each method\'s final unlearned model to '
                             '{MODEL_CHECKPOINT_ROOT}/baseline_models/ (single-experiment mode '
                             'only). Off by default - these runs are normally metrics-only, and '
                             'baseline checkpoints are mainly needed for downstream analysis (e.g. '
                             'a linear-probe evaluation) that not every run requires.')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate a forget-vs-retain t-SNE plot ({name}_{method}_tsne.png) for '
                             'every method run. Off by default - like --compute_ain, this is a '
                             'noticeably more expensive extra step than the other metrics.')

    # --- DELETE arguments ---
    parser.add_argument('--delete_epochs', type=int, default=20,
                        help='Epochs of forget-only masked-logit distillation for the DELETE baseline.')
    parser.add_argument('--delete_lr', type=float, default=1e-4,
                        help='SGD learning rate for the DELETE baseline.')
    parser.add_argument('--delete_disable_bn', action='store_true',
                        help='Freeze BatchNorm running stats during DELETE\'s forget-only fine-tune '
                             '(there\'s no retain-set signal to keep them sane otherwise).')

    # --- SSD arguments ---
    parser.add_argument('--ssd_dampening_constant', type=float, default=1.0,
                        help='SSD dampening constant (lambda).')
    parser.add_argument('--ssd_selection_weighting', type=float, default=None,
                        help='SSD selection weighting (alpha). Default: None, which resolves to '
                             '5.0 for ViT / 10.0 otherwise (matching the reference implementation\'s '
                             'own architecture-aware default).')

    # --- CoUn arguments ---
    parser.add_argument('--coun_epochs', type=int, default=1,
                        help='Epochs of retain-set contrastive training for the CoUn baseline.')
    parser.add_argument('--coun_lr', type=float, default=0.01,
                        help='SGD learning rate for the CoUn baseline.')
    parser.add_argument('--coun_lambda_scale', type=float, default=1.0,
                        help='CoUn\'s contrastive-loss scaling constant (see baselines/coun.py). '
                             'Fixed here, not swept - use the standalone coun.py CLI for the sweep.')
    parser.add_argument('--coun_temp', type=float, default=0.1,
                        help='CoUn\'s contrastive-loss temperature (see baselines/coun.py). '
                             'Fixed here, not swept - use the standalone coun.py CLI for the sweep.')

    # --- CU (Contrastive Unlearning, Lee et al. 2024) arguments ---
    # Not to be confused with CoUn above (a different paper). The paper
    # doesn't state numeric defaults for these - see baselines/cu.py.
    parser.add_argument('--cu_epochs', type=int, default=10,
                        help='Max epochs for the CU baseline (early-stops once forget-set '
                             'accuracy drops to 1/num_classes, per the paper\'s own Algorithm 1).')
    parser.add_argument('--cu_lr', type=float, default=0.01,
                        help='SGD learning rate for the CU baseline.')
    parser.add_argument('--cu_temp', type=float, default=0.1,
                        help='CU\'s contrastive-loss temperature (tau).')
    parser.add_argument('--cu_lambda_ul', type=float, default=1.0,
                        help='CU\'s contrastive unlearning loss weight (lambda_UL).')
    parser.add_argument('--cu_lambda_ce', type=float, default=1.0,
                        help='CU\'s retain-set cross-entropy loss weight (lambda_CE).')
    parser.add_argument('--cu_omega', type=int, default=4,
                        help='CU\'s inner-loop repetitions per forget batch (paper: <= 4).')

    # --- Cheng et al. (retain-forget entanglement) arguments ---
    # Only applicable to data_name values with a known class hierarchy
    # (cifar100/tinyimagenet) - see baselines/cheng_unlearn.py.
    parser.add_argument('--cheng_stage1_epochs', type=int, default=1,
                        help='Stage 1 (augmented-Lagrangian constrained forgetting) epochs.')
    parser.add_argument('--cheng_stage1_lr', type=float, default=2.5e-6,
                        help='Stage 1 Adam learning rate.')
    parser.add_argument('--cheng_mu', type=float, default=10.0,
                        help='Stage 1 augmented-Lagrangian penalty weight (mu).')
    parser.add_argument('--cheng_gamma', type=float, default=1.0,
                        help='Stage 1 raw (unclipped) forget-loss weight (gamma).')
    parser.add_argument('--cheng_c', type=float, default=10.0,
                        help='Stage 1 forget-loss clip value (c).')
    parser.add_argument('--cheng_stage2_epochs', type=int, default=6,
                        help='Stage 2 (W2-regularized gradient-projected fine-tuning) epochs.')
    parser.add_argument('--cheng_stage2_lr', type=float, default=2e-5,
                        help='Stage 2 SGD learning rate.')
    parser.add_argument('--cheng_momentum', type=float, default=0.9,
                        help='Stage 2 SGD momentum.')
    parser.add_argument('--cheng_alpha', type=float, default=0.5,
                        help='Stage 2 W2-penalty blend weight (alpha).')

    # --- SalUn arguments (see baselines/salun.py) ---
    parser.add_argument('--salun_mask_ratio', type=float, default=0.5,
                        help='Fraction of parameters (globally, by |forget-loss gradient|) treated '
                             'as salient and allowed to update during unlearning.')
    parser.add_argument('--salun_epochs', type=int, default=10, help='Random-labeling fine-tuning epochs.')
    parser.add_argument('--salun_lr', type=float, default=0.01, help='SGD learning rate.')
    parser.add_argument('--salun_momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--salun_weight_decay', type=float, default=5e-4, help='SGD weight decay.')
    parser.add_argument('--salun_lr_decay_epochs', type=str, default='5,8',
                        help='Comma-separated epoch numbers at which LR drops by 0.1 (MultiStepLR).')

    # --- Bad Teacher arguments (see baselines/bad_teacher.py) ---
    parser.add_argument('--bt_epochs', type=int, default=1, help='Dual-teacher distillation epochs.')
    parser.add_argument('--bt_lr', type=float, default=1e-4, help='Adam learning rate.')
    parser.add_argument('--bt_temperature', type=float, default=1.0,
                        help='Softmax temperature for both teachers and the student in the KL loss.')

    args, _ = parser.parse_known_args()
    args.salun_lr_decay_epochs = tuple(int(e) for e in args.salun_lr_decay_epochs.split(','))

    BASELINE_DIR = f'{MODEL_CHECKPOINT_ROOT}/baseline_models'

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # --- Single-experiment mode: triggered when --original_model is provided ---
    # Bypasses the hardcoded path_dicts loop entirely.
    if args.original_model is not None:
        single_exp = {
            'data_name':         args.data_name,
            'model_type':        args.model_name,
            'orig_model_path':   args.original_model,
            'retrain_model_path': args.retrain_model,
            'forget_class':      args.forget_class,
            'batch_size':        args.batch_size,
            'custom_unlearn':    False,
            'oculoplastics':     False,
            'data_path':         './data',
        }
        methods_single = [m.strip() for m in args.method.split(',')]
        readouts = {}
        SELECTIVE_UNLEARNING = False
        combined_df = None

        data_name      = single_exp['data_name']
        model_type     = single_exp['model_type']
        orig_model_path    = single_exp['orig_model_path']
        retrain_model_path = single_exp['retrain_model_path']
        forget_class   = single_exp['forget_class']
        batch_size     = single_exp['batch_size']
        custom_unlearn = single_exp['custom_unlearn']
        oculoplastics  = single_exp['oculoplastics']
        data_path      = single_exp['data_path']

        trainset_full, testset, dataset = get_dataset(data_name, data_path, model_name=model_type)

        # Held-out validation split (seeded by --seed), carved off BEFORE any
        # forget/remain splitting - every baseline's own training data
        # (train_remain_loader, full_train_loader, etc.) is built from the
        # remaining (1 - val_fraction) trainset below, never from valset, so
        # no baseline trains on data it's also validated against. Matches
        # main.py/GEAR's own val_fraction convention and default exactly, so
        # the same --seed produces the same train/val partition in both
        # entry points.
        val_size = int(len(trainset_full) * args.val_fraction)
        train_size = len(trainset_full) - val_size
        split_generator = torch.Generator().manual_seed(seed)
        trainset, valset = torch.utils.data.random_split(
            trainset_full, [train_size, val_size], generator=split_generator
        )

        train_loader, test_loader = get_dataloader(trainset, testset, batch_size, device=device)
        num_classes, idx_to_class = set_num_classes(data_name, dataset)
        total_forget_class = sum(1 for _, target in dataset if target == forget_class)
        num_forget = total_forget_class
        print(f"Number to Forget: {num_forget}")

        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict = dataloader_engine(
            batch_size, trainset, testset, combined_df,
            num_forget=num_forget, forget_class=forget_class,
            oculoplastics=oculoplastics, custom_unlearn=custom_unlearn,
            selective_unlearning=SELECTIVE_UNLEARNING)

        final_forget_loader, final_remain_loader = get_forget_loader(testset, forget_class)
        val_forget_loader, val_remain_loader = get_forget_loader(valset, forget_class)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        # SSD needs the full, undivided original trainset (forget+retain
        # combined, not just the retain split) - trainset (built above,
        # before any forget/remain split, but AFTER the val split) is
        # exactly that; it deliberately excludes the held-out validation
        # portion, same as every other baseline's training data.
        full_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        for unlearn_type in methods_single:
            readouts[unlearn_type] = {data_name: {}}
            model = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
            model = load_model_state(model, orig_model_path)

            if unlearn_type == 'scrub':
                print("Forgetting by SCRUB:")
                teacher = model
                student = model
                target_forget_acc = load_retrain_forget_acc(
                    retrain_model_path, model_type, num_classes, data_name, train_forget_loader, device
                )
                _t0 = time.time()
                model_s, model_s_final = scrub_unlearn(
                    teacher, student, train_remain_loader, train_forget_loader, model_type, data_name,
                    sgda_epochs=args.scrub_epochs,
                    feature_contrastive=args.feature_contrastive,
                    use_entanglement_weighting=args.use_entanglement_weighting,
                    target_layer=args.target_layer,
                    retain_forget_weight=args.retain_forget_weight,
                    forget_forget_weight=args.forget_forget_weight,
                    feature_align_weight=args.feature_align_weight,
                    gamma_rep=args.gamma_rep,
                    remain_reg=args.remain_reg,
                    centroid_refresh_interval=args.centroid_refresh_interval,
                    beta_ce=args.beta_ce,
                    num_classes=num_classes,
                    target_forget_acc=target_forget_acc,
                )
                # SCRUB-R and SCRUB both come out of this one combined
                # training run - splitting the elapsed time between them
                # wouldn't mean anything, so both report the same run time.
                _scrub_time = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_s, 'scrub-r', model_type, data_name, forget_class, seed, BASELINE_DIR)
                    save_baseline_checkpoint(model_s_final, 'scrub', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = {
                    "SCRUB-R": all_readouts(model_s, test_loader, final_forget_loader, final_remain_loader, name='SCRUB-R', seed=seed, unlearn_time=_scrub_time),
                    "SCRUB":   all_readouts(model_s_final, test_loader, final_forget_loader, final_remain_loader, name='SCRUB', seed=seed, unlearn_time=_scrub_time),
                }
            elif unlearn_type == 'finetune':
                print("Forgetting by Fine-tuning:")
                _t0 = time.time()
                finetune(model, train_remain_loader, epochs=10, quiet=True, lr=0.04)
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model, 'finetune', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model, test_loader, final_forget_loader, final_remain_loader, name='Finetune', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'neggrad':
                print("Forgetting by NegGrad:")
                _t0 = time.time()
                negative_grad(model, train_remain_loader, train_forget_loader, alpha=0.9999, epochs=5, quiet=True, lr=0.01)
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model, 'neggrad', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model, test_loader, final_forget_loader, final_remain_loader, name='NegGrad', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'cfk':
                if model_type not in CFK_EUK_SUPPORTED_MODELS:
                    print(f"[cfk] Not applicable to model_name='{model_type}' - CFK's 'freeze all but "
                          f"the last representational block' logic has no branch for this architecture "
                          f"(see baselines/euk.py:CFK_EUK_SUPPORTED_MODELS). Skipping.")
                else:
                    print("Forgetting by CFK:")
                    _t0 = time.time()
                    model_cfk = cfk_unlearn(model, train_remain_loader, model_type)
                    _elapsed = time.time() - _t0
                    if args.save_checkpoints:
                        save_baseline_checkpoint(model_cfk, 'cfk', model_type, data_name, forget_class, seed, BASELINE_DIR)
                    readouts[unlearn_type][data_name] = all_readouts(model_cfk, test_loader, final_forget_loader, final_remain_loader, name='CFK', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'euk':
                if model_type not in CFK_EUK_SUPPORTED_MODELS:
                    print(f"[euk] Not applicable to model_name='{model_type}' - EUK's 'reset then "
                          f"fine-tune the last representational block' logic has no branch for this "
                          f"architecture (see baselines/euk.py:CFK_EUK_SUPPORTED_MODELS). Skipping.")
                else:
                    print("Forgetting by EUK:")
                    _t0 = time.time()
                    model_euk = euk_unlearn(model, train_remain_loader, model_type)
                    _elapsed = time.time() - _t0
                    if args.save_checkpoints:
                        save_baseline_checkpoint(model_euk, 'euk', model_type, data_name, forget_class, seed, BASELINE_DIR)
                    readouts[unlearn_type][data_name] = all_readouts(model_euk, test_loader, final_forget_loader, final_remain_loader, name='EUK', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'delete':
                print("Forgetting by DELETE:")
                _t0 = time.time()
                model_delete = delete_unlearn(
                    model, train_forget_loader, device,
                    unlearn_epoch=args.delete_epochs, unlearn_rate=args.delete_lr,
                    disable_bn=args.delete_disable_bn,
                )
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_delete, 'delete', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model_delete, test_loader, final_forget_loader, final_remain_loader, name='DELETE', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'ssd':
                print("Forgetting by SSD:")
                _t0 = time.time()
                model_ssd = ssd_unlearn(
                    model, train_forget_loader, full_train_loader, device,
                    dampening_constant=args.ssd_dampening_constant,
                    selection_weighting=args.ssd_selection_weighting,
                    model_name=model_type,
                )
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_ssd, 'ssd', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model_ssd, test_loader, final_forget_loader, final_remain_loader, name='SSD', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'coun':
                if model_type not in COUN_VALID_PAIRINGS.get(data_name, []):
                    print(f"[coun] Not applicable to data_name='{data_name}'/model_name='{model_type}' - "
                          f"CoUn's core update uses SimCLR-style image augmentations, which don't apply to "
                          f"text. Skipping.")
                else:
                    print("Forgetting by CoUn:")
                    _, _, trainset_coun_raw = get_coun_datasets(data_name, model_type, data_path)
                    train_remain_loader_raw = DataLoader(trainset_coun_raw, batch_size=batch_size,
                                                         sampler=SubsetRandomSampler(train_remain_index))
                    _t0 = time.time()
                    model_coun = coun_unlearn(
                        model, model_type, data_name, train_remain_loader_raw, device,
                        lambda_scale=args.coun_lambda_scale, temp=args.coun_temp,
                        epochs=args.coun_epochs, lr=args.coun_lr,
                    )
                    _elapsed = time.time() - _t0
                    if args.save_checkpoints:
                        save_baseline_checkpoint(model_coun, 'coun', model_type, data_name, forget_class, seed, BASELINE_DIR)
                    readouts[unlearn_type][data_name] = all_readouts(model_coun, test_loader, final_forget_loader, final_remain_loader, name='CoUn', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'cu':
                print("Forgetting by CU:")
                _t0 = time.time()
                model_cu = cu_unlearn(
                    model, train_forget_loader, train_remain_loader, device, num_classes,
                    lambda_ul=args.cu_lambda_ul, lambda_ce=args.cu_lambda_ce, temperature=args.cu_temp,
                    omega=args.cu_omega, lr=args.cu_lr, max_epochs=args.cu_epochs,
                    eval_forget_loader=final_forget_loader,
                )
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_cu, 'cu', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model_cu, test_loader, final_forget_loader, final_remain_loader, name='CU', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'cheng_unlearn':
                print("Forgetting by Cheng et al. (retain-forget entanglement):")
                adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(data_name, forget_class, trainset)
                if adjacent_indices is None or remote_indices is None:
                    print(f"[cheng_unlearn] No class hierarchy for data_name='{data_name}' - this baseline "
                          f"needs Retain Adjacent/Remote splits as a training input, not just an eval metric. Skipping.")
                else:
                    adjacent_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(adjacent_indices))
                    remote_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(remote_indices))
                    _t0 = time.time()
                    model_cheng = cheng_unlearn(
                        model, train_forget_loader, adjacent_loader, remote_loader, device,
                        stage1_epochs=args.cheng_stage1_epochs, stage1_lr=args.cheng_stage1_lr,
                        mu=args.cheng_mu, gamma=args.cheng_gamma, c=args.cheng_c,
                        stage2_epochs=args.cheng_stage2_epochs, stage2_lr=args.cheng_stage2_lr,
                        momentum=args.cheng_momentum, alpha=args.cheng_alpha,
                    )
                    _elapsed = time.time() - _t0
                    if args.save_checkpoints:
                        save_baseline_checkpoint(model_cheng, 'cheng_unlearn', model_type, data_name, forget_class, seed, BASELINE_DIR)
                    readouts[unlearn_type][data_name] = all_readouts(model_cheng, test_loader, final_forget_loader, final_remain_loader, name='ChengUnlearn', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'salun':
                print("Forgetting by SalUn:")
                _t0 = time.time()
                model_salun = salun_unlearn(
                    model, train_forget_loader, train_remain_loader, device, num_classes,
                    mask_ratio=args.salun_mask_ratio, unlearn_epochs=args.salun_epochs,
                    lr=args.salun_lr, momentum=args.salun_momentum, weight_decay=args.salun_weight_decay,
                    lr_decay_epochs=args.salun_lr_decay_epochs,
                )
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_salun, 'salun', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model_salun, test_loader, final_forget_loader, final_remain_loader, name='SalUn', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'bad_teacher':
                print("Forgetting by Bad Teacher:")
                _t0 = time.time()
                model_bt = bad_teacher_unlearn(
                    model, train_forget_loader, train_remain_loader, model_type, num_classes, data_name, device,
                    epochs=args.bt_epochs, lr=args.bt_lr, temperature=args.bt_temperature,
                )
                _elapsed = time.time() - _t0
                if args.save_checkpoints:
                    save_baseline_checkpoint(model_bt, 'bad_teacher', model_type, data_name, forget_class, seed, BASELINE_DIR)
                readouts[unlearn_type][data_name] = all_readouts(model_bt, test_loader, final_forget_loader, final_remain_loader, name='BadTeacher', seed=seed, unlearn_time=_elapsed)
            elif unlearn_type == 'eval_orig':
                # Not an unlearning method - evaluates --retrain_model itself
                # as the gold-standard reference row. No unlearning happens,
                # so Run Time is correctly 'N/A' (categorically inapplicable,
                # not merely unmeasured) rather than a timed value.
                print("Evaluating Retrain Model:")
                model0 = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                model0 = load_model_state(model0, retrain_model_path)
                readouts[unlearn_type][data_name] = {
                    "Retrain": all_readouts(model0, test_loader, final_forget_loader, final_remain_loader, name='Retrain', seed=seed)
                }
            elif unlearn_type == 'eval_original':
                # Mirrors eval_orig exactly, but for --original_model - the
                # pre-unlearning model, as its own first-class reportable
                # configuration (not just an input other methods load and
                # then modify). Same 'N/A' Run Time reasoning as eval_orig.
                print("Evaluating Original Model:")
                model_orig_eval = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                model_orig_eval = load_model_state(model_orig_eval, orig_model_path)
                readouts[unlearn_type][data_name] = {
                    "Original": all_readouts(model_orig_eval, test_loader, final_forget_loader, final_remain_loader, name='Original', seed=seed)
                }
            else:
                print(f"Method '{unlearn_type}' not supported in single-experiment mode.")

        import json
        output_file = f"{args.name}_{data_name}_results.json"
        with open(output_file, "w") as f:
            json.dump(readouts, f, indent=4)
        print(f"Results saved to {output_file}")
        import sys; sys.exit(0)
    # --- End single-experiment mode ---

    retain_bs = 32
    forget_bs = 16
    batch_size = 8

    methods = ['finetune', 'neggrad', 'cfk', 'euk', 'scrub', 'delete', 'ssd', 'coun', 'cu', 'cheng_unlearn', 'ravi', 'chen','eval_orig']

    SELECTIVE_UNLEARNING = False
    oculoplastics =  False
    med_unlearn = True
    combined_df = None

    if SELECTIVE_UNLEARNING and not med_unlearn:
        model_paths = selective_forget_models
    if med_unlearn:
        model_paths = med_unlearn_paths

    for i, (data_name, model_list) in enumerate(model_paths.items()):
        if data_name == 'mri':
            percentages = [.1, .25, .5, .75]
        elif data_name == 'fashionmnist':
            percentages = [.01, .1, .25, .5, .75]
        else:
            percentages = [1]
        for unlearn_type in methods:
            readouts = {}

            for percentage in percentages:

                print(f"Iteration {i}: {data_name} -> {model_list}")

                if SELECTIVE_UNLEARNING == False:
                    percentage = 1
                    if unlearn_type not in readouts:
                        readouts[unlearn_type] = {}
                    if data_name not in readouts[unlearn_type]:
                        readouts[unlearn_type][data_name] = {}

                    orig_model_path = model_list[0]
                    retrain_model_path = model_list[1]
                    model_type = model_list[2]

                    if med_unlearn == True:
                        forget_class = int(model_list[3])
                        custom_unlearn = model_list[4]


                else:
                    print('💀BEWARE: MAKE SURE YOU REALLY WANT TO DO THIS💀')
                    if unlearn_type not in readouts:
                        readouts[unlearn_type] = {}
                    if data_name not in readouts[unlearn_type]:
                        readouts[unlearn_type][data_name] = {}
                    if str(percentage) not in readouts[unlearn_type][data_name]:
                        readouts[unlearn_type][data_name][str(percentage)] = {}

                    orig_model_path = model_list['original']
                    retrain_model_path = model_list['retrain'][str(percentage)]
                    model_type = model_list['model_type']


                if data_name == 'open_source':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/fundus_open_source'

                elif data_name == 'mri':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/mri_unlearn'

                elif data_name == 'ultrasound':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/ultrasound_unlearn_oversample'

                elif data_name == 'oct_4_class':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/oct_open_source'

                elif data_name == 'oculoplastic':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/oculoplastic'
                    oculoplastics =True
                    if custom_unlearn:
                        ted_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_oculoplastic/mm_07022024_full_run_TED_GT_pix.csv')
                        cfd_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_oculoplastic/mm_07022024_full_run_CFD_GT_pix.csv')
                        combined_df = pd.concat([ted_df, cfd_df], ignore_index=True)

                elif data_name == 'fundus_3_class':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/fundus_big'
                    if custom_unlearn:
                        ord_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Other_Retinal_Disorders_UNIQUE_MRN_filtered.csv')
                        dr_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Diabetic_Retinopathy_UNIQUE_MRN_filtered.csv')
                        glauc_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Glaucoma_UNIQUE_MRN_filtered.csv')
                        combined_df = pd.concat([ord_df, dr_df, glauc_df], ignore_index=True)

                else:
                    data_path = './data'

                print(f'EXPERIMENTAL REPORT: \ncustom unlearn :  {custom_unlearn}, \n data name : {data_name} \n oculoplastics : {oculoplastics} \n forget class : {forget_class} \n unlearn type : {unlearn_type} ')

                trainset, testset, dataset = get_dataset(data_name, data_path, model_name=model_type)
                train_loader, test_loader = get_dataloader(trainset, testset, batch_size, device=device)
                # SSD needs the full, undivided original trainset (forget+retain
                # combined, not just the retain split) - trainset (built just
                # above, before any forget/remain split) is exactly that.
                full_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

                # set number of classes
                num_classes, idx_to_class = set_num_classes(data_name, dataset)
                total_forget_class = sum(1 for _, target in dataset if target == forget_class)
                num_forget = int(total_forget_class * percentage)
                print(f"Forget Percentage: {percentage}, Number to Forget: {num_forget}")



                train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
                train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict = dataloader_engine(batch_size, trainset, testset,
                                                                                                                combined_df, num_forget=num_forget, forget_class = forget_class,
                                                                                                                oculoplastics=oculoplastics, custom_unlearn = custom_unlearn,
                                                                                                                selective_unlearning = SELECTIVE_UNLEARNING)


                if SELECTIVE_UNLEARNING and not med_unlearn:
                    final_forget_loader = train_forget_loader
                    final_remain_loader = train_remain_loader
                    assert set(train_forget_index) == set(test_forget_index), "Train and test forget indices do not match!"
                    assert set(train_remain_index) == set(test_remain_index), "Train and test remain indices do not match!"
                else:
                    if not custom_unlearn:
                        if SELECTIVE_UNLEARNING:
                            final_forget_loader = train_forget_loader
                            final_remain_loader = train_remain_loader
                        else:
                            final_forget_loader, final_remain_loader = get_forget_loader(testset, forget_class)

                    elif custom_unlearn and not oculoplastics:
                        final_forget_loader, final_remain_loader = get_custom_forget_loader(testset, test_dict, 'Cirrus 800 FA')

                    elif custom_unlearn and oculoplastics:
                        final_forget_loader, final_remain_loader = get_custom_forget_loader_oculoplastics(testset, test_dict)

                model = load_model(model_type, num_classes=num_classes, data_name =data_name).to(device)
                model = load_model_state(model, orig_model_path)

                if unlearn_type == 'finetune':
                    print("Forgetting by Fine-tuning:")
                    ft_lr = 0.04
                    model_ft = model
                    ft_epochs = 10

                    finetune(model_ft, train_remain_loader, epochs=ft_epochs, quiet=True, lr=ft_lr)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ft, test_loader, final_forget_loader, final_remain_loader, name='Finetune', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ft, test_loader, final_forget_loader, final_remain_loader, name='Finetune', seed=seed)

                elif unlearn_type == 'neggrad':
                    print("Forgetting by NegGrad:")
                    model_ng = model
                    ng_alpha = 0.9999
                    ng_epochs = 5
                    ng_lr = 0.01
                    negative_grad(model_ng, train_remain_loader, train_forget_loader, alpha=ng_alpha, epochs=ng_epochs, quiet=True, lr=ng_lr)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ng, test_loader, final_forget_loader, final_remain_loader, name='NegGrad', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ng, test_loader, final_forget_loader, final_remain_loader, name='NegGrad', seed=seed)

                elif unlearn_type == 'cfk':
                    print("Forgetting by CFK:")
                    model_cfk = cfk_unlearn(model, train_remain_loader, model_type)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_cfk, test_loader, final_forget_loader, final_remain_loader, name='CFK', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_cfk, test_loader, final_forget_loader, final_remain_loader, name='CFK', seed=seed)

                elif unlearn_type == 'euk':
                    print("Forgetting by EUK:")
                    model_initial = model
                    model_euk = euk_unlearn(model, train_remain_loader, model_type)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_euk, test_loader, final_forget_loader, final_remain_loader, name='EUK', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_euk, test_loader, final_forget_loader, final_remain_loader, name='EUK', seed=seed)

                elif unlearn_type == 'delete':
                    print("Forgetting by DELETE:")
                    model_delete = delete_unlearn(
                        model, train_forget_loader, device,
                        unlearn_epoch=args.delete_epochs, unlearn_rate=args.delete_lr,
                        disable_bn=args.delete_disable_bn,
                    )
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_delete, test_loader, final_forget_loader, final_remain_loader, name='DELETE', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_delete, test_loader, final_forget_loader, final_remain_loader, name='DELETE', seed=seed)

                elif unlearn_type == 'ssd':
                    print("Forgetting by SSD:")
                    model_ssd = ssd_unlearn(
                        model, train_forget_loader, full_train_loader, device,
                        dampening_constant=args.ssd_dampening_constant,
                        selection_weighting=args.ssd_selection_weighting,
                        model_name=model_type,
                    )
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ssd, test_loader, final_forget_loader, final_remain_loader, name='SSD', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ssd, test_loader, final_forget_loader, final_remain_loader, name='SSD', seed=seed)

                elif unlearn_type == 'coun':
                    if model_type not in COUN_VALID_PAIRINGS.get(data_name, []):
                        print(f"[coun] Not applicable to data_name='{data_name}'/model_name='{model_type}' - "
                              f"CoUn's core update uses SimCLR-style image augmentations, which don't apply to "
                              f"text. Skipping.")
                    else:
                        print("Forgetting by CoUn:")
                        _, _, trainset_coun_raw = get_coun_datasets(data_name, model_type, data_path)
                        train_remain_loader_raw = DataLoader(trainset_coun_raw, batch_size=batch_size,
                                                             sampler=SubsetRandomSampler(train_remain_index))
                        model_coun = coun_unlearn(
                            model, model_type, data_name, train_remain_loader_raw, device,
                            lambda_scale=args.coun_lambda_scale, temp=args.coun_temp,
                            epochs=args.coun_epochs, lr=args.coun_lr,
                        )
                        if not SELECTIVE_UNLEARNING:
                            readouts[unlearn_type][data_name] = all_readouts(model_coun, test_loader, final_forget_loader, final_remain_loader, name='CoUn', seed=seed)
                        else:
                            readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_coun, test_loader, final_forget_loader, final_remain_loader, name='CoUn', seed=seed)

                elif unlearn_type == 'cu':
                    print("Forgetting by CU:")
                    model_cu = cu_unlearn(
                        model, train_forget_loader, train_remain_loader, device, num_classes,
                        lambda_ul=args.cu_lambda_ul, lambda_ce=args.cu_lambda_ce, temperature=args.cu_temp,
                        omega=args.cu_omega, lr=args.cu_lr, max_epochs=args.cu_epochs,
                        eval_forget_loader=final_forget_loader,
                    )
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_cu, test_loader, final_forget_loader, final_remain_loader, name='CU', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_cu, test_loader, final_forget_loader, final_remain_loader, name='CU', seed=seed)

                elif unlearn_type == 'cheng_unlearn':
                    print("Forgetting by Cheng et al. (retain-forget entanglement):")
                    adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(data_name, forget_class, trainset)
                    if adjacent_indices is None or remote_indices is None:
                        print(f"[cheng_unlearn] No class hierarchy for data_name='{data_name}' - this baseline "
                              f"needs Retain Adjacent/Remote splits as a training input, not just an eval metric. Skipping.")
                    else:
                        adjacent_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(adjacent_indices))
                        remote_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(remote_indices))
                        model_cheng = cheng_unlearn(
                            model, train_forget_loader, adjacent_loader, remote_loader, device,
                            stage1_epochs=args.cheng_stage1_epochs, stage1_lr=args.cheng_stage1_lr,
                            mu=args.cheng_mu, gamma=args.cheng_gamma, c=args.cheng_c,
                            stage2_epochs=args.cheng_stage2_epochs, stage2_lr=args.cheng_stage2_lr,
                            momentum=args.cheng_momentum, alpha=args.cheng_alpha,
                        )
                        if not SELECTIVE_UNLEARNING:
                            readouts[unlearn_type][data_name] = all_readouts(model_cheng, test_loader, final_forget_loader, final_remain_loader, name='ChengUnlearn', seed=seed)
                        else:
                            readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_cheng, test_loader, final_forget_loader, final_remain_loader, name='ChengUnlearn', seed=seed)

                elif unlearn_type == 'scrub':
                    print("Forgetting by SCRUB:")
                    teacher = model
                    student = model

                    target_forget_acc = load_retrain_forget_acc(
                        retrain_model_path, model_type, num_classes, data_name, train_forget_loader, device
                    )
                    model_s, model_s_final = scrub_unlearn(
                        teacher, student, train_remain_loader, train_forget_loader, model_type, data_name,
                        sgda_epochs=args.scrub_epochs,
                        feature_contrastive=args.feature_contrastive,
                        use_entanglement_weighting=args.use_entanglement_weighting,
                        target_layer=args.target_layer,
                        retain_forget_weight=args.retain_forget_weight,
                        forget_forget_weight=args.forget_forget_weight,
                        feature_align_weight=args.feature_align_weight,
                        gamma_rep=args.gamma_rep,
                        remain_reg=args.remain_reg,
                        centroid_refresh_interval=args.centroid_refresh_interval,
                        beta_ce=args.beta_ce,
                        num_classes=num_classes,
                        target_forget_acc=target_forget_acc,
                    )
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = {
                            "SCRUB-R": all_readouts(model_s, test_loader, final_forget_loader, final_remain_loader, name='SCRUB-R', seed=seed),
                            "SCRUB": all_readouts(model_s_final, test_loader, final_forget_loader, final_remain_loader, name='SCRUB', seed=seed)
                        }
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = {
                            "SCRUB-R": all_readouts(model_s, test_loader, final_forget_loader, final_remain_loader, name='SCRUB-R', seed=seed),
                            "SCRUB": all_readouts(model_s_final, test_loader, final_forget_loader, final_remain_loader, name='SCRUB', seed=seed)
                        }

                elif unlearn_type == 'chen':
                    print("Forgetting by Chen et al.:")
                    model_chen = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    if SELECTIVE_UNLEARNING:
                        chen_ckpt = model_list['chen'][str(percentage)]
                    else:
                        if med_unlearn and custom_unlearn:
                            chen_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][1])
                        if med_unlearn and not custom_unlearn:
                            chen_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][0])

                    print(f"Loading Chen baseline from {chen_ckpt}")
                    model_chen = load_checkpoint_without_dataparallel(chen_ckpt, model_chen)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_chen, test_loader, final_forget_loader, final_remain_loader, name='Chen', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_chen, test_loader, final_forget_loader, final_remain_loader, name='Chen', seed=seed)

                elif unlearn_type == 'ravi':
                    print("Forgetting by Ravi et al.:")
                    model_ravi = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    if SELECTIVE_UNLEARNING:
                        ravi_ckpt = model_list['ravi'][str(percentage)]
                    else:
                        if med_unlearn and custom_unlearn:
                            ravi_ckpt = os.path.join(BASELINE_DIR, ravi_paths[data_name][1])
                        if med_unlearn and not custom_unlearn:
                            ravi_ckpt = os.path.join(BASELINE_DIR, ravi_paths[data_name][0])

                    print(f"Loading Ravi baseline from {ravi_ckpt}")
                    model_ravi = load_checkpoint_without_dataparallel(ravi_ckpt, model_ravi)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ravi, test_loader, final_forget_loader, final_remain_loader, name='Ravi', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ravi, test_loader, final_forget_loader, final_remain_loader, name='Ravi', seed=seed)

                elif unlearn_type == 'eval_orig':
                    print("Evaluating Original and Retrain Models:")
                    model0 = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    model0 = load_model_state(model0, retrain_model_path)

                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = {
                            "Retrain": all_readouts(model0, test_loader, final_forget_loader, final_remain_loader, name='Retrain', seed=seed)
                        }
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = {
                            "Retrain": all_readouts(model0, test_loader, final_forget_loader, final_remain_loader, name='Retrain', seed=seed)
                        }


            import json
            output_file = f"med_unlearn_{data_name}_{unlearn_type}_{custom_unlearn}_{forget_class}.json"
            with open(output_file, "w") as f:
                json.dump(readouts, f, indent=4)
            custom_unlearn= False
            oculoplastics =  False

            print(f"Results saved to {output_file}")
