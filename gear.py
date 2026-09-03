import copy
from trainer import eval
import numpy as np
import torch
from torch import nn
import tqdm
import time
from make_dataloaders import *
import class_hierarchy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def inf_generator(iterable):
    """Wraps a DataLoader so it can be iterated forever, re-starting from the
    beginning whenever it's exhausted. Used so the training loop below can
    run for exactly num_iterations steps regardless of how long the forget
    and remain loaders happen to be relative to each other."""
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# =============================================================================
# FEATURE EXTRACTION + CONTRASTIVE LOSS HELPERS
# =============================================================================

RESNET_ALL_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']


def get_intermediate_features(model, x, target_layer):
    """Runs model.forward_with_features and returns just the requested
    layer's activation. AllCNN's layers are addressed by integer index
    ("9" -> 9); ResNet-50's are addressed by name ("layer4")."""
    if isinstance(target_layer, str) and target_layer.isdigit():
        target_layer = int(target_layer)
    if hasattr(model, 'module'):
        _, feature_dict = model.module.forward_with_features(x, capture_layers=[target_layer])
    else:
        _, feature_dict = model.forward_with_features(x, capture_layers=[target_layer])
    return feature_dict[target_layer]

def get_intermediate_features_multilayer(model, x, layers):
    """Returns a dict of {layer_name: feature_tensor} for each layer in layers."""
    m = model.module if hasattr(model, 'module') else model
    _, feature_dict = m.forward_with_features(x, capture_layers=layers)
    return feature_dict


def normalize_rows(x, eps=1e-8):
    """L2-normalizes each row of a [N, D] tensor, epsilon-stabilized."""
    return x / (x.norm(dim=1, keepdim=True) + eps)


def retain_alignment_loss(current_feats, ref_feats):
    """1 - mean cosine similarity to the frozen reference (original) model's
    features at the same layer. Penalizes retain-set representations for
    drifting away from where the pristine original model put them."""
    current_feats = normalize_rows(current_feats)
    ref_feats = normalize_rows(ref_feats)
    return 1 - (current_feats * ref_feats).sum(dim=1).mean()


def retain_forget_loss(retain_feats, forget_feats, entanglement_scores=None):
    """Mean cosine similarity between retain and forget features. Minimizing
    this pushes forget-class representations away from retain-class ones. If
    entanglement_scores are given, each forget sample's contribution is
    scaled by how entangled it currently is with the retain distribution, so
    the most entangled samples get pushed hardest."""
    retain_feats = normalize_rows(retain_feats)
    forget_feats = normalize_rows(forget_feats)
    sim = torch.einsum('md,nd->mn', retain_feats, forget_feats)  # [M, N]
    if entanglement_scores is not None:
        # Scale each forget sample's column by its entanglement score, then average
        weighted = sim * entanglement_scores.unsqueeze(0)  # [M, N]
        return weighted.mean()
    return sim.mean()


def forget_forget_loss(forget_feats):
    """Negative mean pairwise cosine similarity among forget samples in the
    batch (diagonal/self-similarity excluded). Since this is minimized, it
    actually increases pairwise similarity — it pulls forget samples toward
    a shared region of feature space, rather than dispersing them."""
    forget_feats = normalize_rows(forget_feats)
    sim = torch.einsum('md,nd->mn', forget_feats, forget_feats)
    upper = torch.triu(sim, diagonal=1)
    n = forget_feats.size(0)
    num_pairs = max(n * (n - 1) / 2, 1)
    return -upper.sum() / num_pairs


def prepare_features(feat):
    """Turns a raw feature map into a comparable L2-normalized vector:
    global-average-pools conv feature maps down to one vector per sample,
    flattens anything else, then L2-normalizes."""
    if feat.dim() == 4:
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.flatten(1)
    elif feat.dim() > 2:
        feat = feat.flatten(1)
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def _get_primary_layer(target_layer):
    """Return the single layer used for centroid / entanglement computation.

    When target_layer == 'all' (multi-layer ResNet mode) we pin to the
    deepest ResNet block so the centroid lives in the most semantic space.
    """
    return RESNET_ALL_LAYERS[-1] if target_layer == 'all' else target_layer


def _resolve_deepest_layer(model, candidate_layers):
    """Dynamically find the deepest layer from candidate_layers that exists
    as a named child of the model, unwrapping DataParallel if present.

    Walking the candidate list in reverse and returning the first hit means
    this works regardless of whether the architecture exposes all candidates.
    Falls back to the last candidate if nothing matches.
    """
    m = model.module if hasattr(model, 'module') else model
    named_children = {name for name, _ in m.named_children()}
    for layer in reversed(candidate_layers):
        if str(layer) in named_children:
            return layer
    # Fallback: no candidate found — return the last one
    return candidate_layers[-1]


@torch.no_grad()
def compute_retain_centroids(model, retain_loader, primary_layer, num_classes, device):
    """Full no-grad pass over retain_loader → per-class L2-normalised centroids.

    Returns a [num_classes, feat_dim] tensor stored as a plain buffer (not a
    Parameter). Classes that have no retain samples get a zero centroid after
    normalisation (handled by clamping the denominator to 1).
    """
    was_training = model.training
    model.eval()

    sum_feats = None
    counts = torch.zeros(num_classes, device=device)

    for x_r, y_r in retain_loader:
        x_r, y_r = x_r.to(device), y_r.to(device)
        feats = get_intermediate_features(model, x_r, primary_layer)
        feats = prepare_features(feats)  # [B, D], already L2-normalised

        if sum_feats is None:
            sum_feats = torch.zeros(num_classes, feats.size(1), device=device)

        for c in range(num_classes):
            mask = (y_r == c)
            if mask.any():
                sum_feats[c] += feats[mask].sum(dim=0)
                counts[c] += mask.sum().float()

    if sum_feats is None:
        raise RuntimeError("retain_loader is empty — cannot compute centroids.")

    # Per-class mean; clamp denominator so empty classes don't cause div-by-zero
    centroids = sum_feats / counts.clamp(min=1.0).unsqueeze(1)
    centroids = F.normalize(centroids, p=2, dim=1)

    if was_training:
        model.train()

    return centroids  # [num_classes, feat_dim]


def compute_entanglement_scores(forget_feats_normed, centroids):
    """e_i = max_c cos(z_i^f, μ_c^r)  for each forget sample i — the highest
    cosine similarity between that sample's features and any retain class's
    centroid. High e_i means the forget sample currently looks a lot like
    some retain class (heavily "entangled"); low e_i means it's already
    well-separated from all retain classes.

    Args:
        forget_feats_normed: [B_f, D] L2-normalised forget features
        centroids:           [num_classes, D] L2-normalised retain centroids
    Returns:
        e: [B_f] entanglement score per forget sample (in [-1, 1])
    """
    sims = torch.mm(forget_feats_normed, centroids.t())  # [B_f, num_classes]
    e, _ = sims.max(dim=1)
    return e


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def _retain_metrics_single_layer(original_model, unlearn_model, data_loader, device, layer):
    """Returns (cos_sim, l1_diff) for a single layer."""
    original_model.eval()
    unlearn_model.eval()
    total_cos = 0.0
    total_diff = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            orig_feats = get_intermediate_features(original_model, batch_x, layer)
            unlearn_feats = get_intermediate_features(unlearn_model, batch_x, layer)
            # cosine similarity
            orig_norm = prepare_features(orig_feats)
            unlearn_norm = prepare_features(unlearn_feats)
            cos = (orig_norm * unlearn_norm).sum(dim=1)
            total_cos += cos.sum().item()
            # L1 diff (pool spatial dims if needed)
            if orig_feats.dim() == 4:
                orig_feats = F.adaptive_avg_pool2d(orig_feats, 1).flatten(1)
                unlearn_feats = F.adaptive_avg_pool2d(unlearn_feats, 1).flatten(1)
            diff = (orig_feats - unlearn_feats).abs().mean(dim=1)
            total_diff += diff.sum().item()
            total_count += diff.size(0)
    return total_cos / total_count, total_diff / total_count


def retain_intermediate_cosine_similarity(original_model, unlearn_model, data_loader, device, target_layer):
    """
    Mean cosine similarity of intermediate features at target_layer between
    original and unlearned model on the retain set.
    When target_layer == 'all', averages over RESNET_ALL_LAYERS.
    Higher = more stable representations (closer to original).
    """
    if target_layer == 'all':
        cos_sims = [
            _retain_metrics_single_layer(original_model, unlearn_model, data_loader, device, l)[0]
            for l in RESNET_ALL_LAYERS
        ]
        return float(np.mean(cos_sims))
    return _retain_metrics_single_layer(original_model, unlearn_model, data_loader, device, target_layer)[0]


def retain_intermediate_feature_l1_diff(original_model, unlearn_model, data_loader, device, target_layer):
    """
    Mean L1 drift of intermediate features at target_layer between
    original and unlearned model on the retain set.
    When target_layer == 'all', averages over RESNET_ALL_LAYERS.
    Lower = less representation shift (better stability).
    """
    if target_layer == 'all':
        l1_diffs = [
            _retain_metrics_single_layer(original_model, unlearn_model, data_loader, device, l)[1]
            for l in RESNET_ALL_LAYERS
        ]
        return float(np.mean(l1_diffs))
    return _retain_metrics_single_layer(original_model, unlearn_model, data_loader, device, target_layer)[1]



def compute_mia(model, forget_loader, test_loader, device, n_splits=5):
    """Confidence-vector membership-inference attack: trains a logistic-
    regression classifier to distinguish forget-set (member) samples from
    test-set (non-member) samples by the shape of their softmax output.
    Returns (mean, std) balanced accuracy across cross-validation folds.
    Near-chance accuracy (~0.5) means the model's behavior on forgotten
    samples is indistinguishable from unseen data — the desired outcome."""
    model.eval()

    def get_confidence_scores(loader):
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
            # Use full probability vector, not just max
                scores.append(probs.cpu().numpy())
        return np.vstack(scores)

    test_scores = get_confidence_scores(test_loader)
    forget_scores = get_confidence_scores(forget_loader)

    n = min(len(forget_scores), len(test_scores))
    rng = np.random.default_rng(42)
    forget_scores = forget_scores[rng.choice(len(forget_scores), n, replace=False)]
    test_scores   = test_scores[rng.choice(len(test_scores),   n, replace=False)]



    X = np.concatenate([forget_scores, test_scores])
    y= np.concatenate([np.ones(n), np.zeros(n)])

    clf = LogisticRegression(max_iter = 1000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        fold_accs.append(balanced_accuracy_score(y[test_idx], preds))

    return float(np.mean(fold_accs)), float(np.std(fold_accs))


def compute_loss_threshold_mia(model, member_loader, nonmember_loader, device):
    """
    Standard loss-threshold MIA for unlearning evaluation.

    Attack: For each sample, compute cross-entropy loss under the model.
    Lower loss suggests the sample was a training member.

    Args:
        model: model to audit
        member_loader: samples the model was trained on (e.g. train_forget_loader)
        nonmember_loader: samples never seen during training (e.g. test_forget_loader)
        device: torch device

    Returns:
        dict with keys:
          'mia_auc': AUC of attack (0.5 = random, 1.0 = perfect)
          'mia_acc': best-threshold balanced accuracy (0.5 = random)
    """
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

    # Negate losses: lower loss → more likely member → higher score
    mia_auc = roc_auc_score(labels, -losses)

    # Best-threshold balanced accuracy
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
# PLOTTING
# =============================================================================

def _plot_loss_curves(loss_log: dict, output_name: str) -> None:
    """
    Plot each GEAR loss component across training iterations.
    Saves to {output_name}_loss_curves.png.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    keys = ["retain_loss", "contrastive_loss", "entanglement_score", "total_loss"]
    colors = ["blue", "green", "purple", "black"]
    titles = [
        "Retain Loss (β · remain_loss)",
        "Contrastive Loss (γ_rep · rep_loss)",
        "Mean Entanglement Score",
        "Total Loss"
    ]

    for ax, key, color, title in zip(axes, keys, colors, titles):
        ax.plot(loss_log[key], color=color, linewidth=1.0, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss" if key != "entanglement_score" else "Score")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"GEAR Loss Curves: {os.path.basename(output_name)}", fontsize=12)
    plt.tight_layout()
    save_path = f"{output_name}_loss_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {save_path}")


# =============================================================================
# RESULTS LOGGING
# =============================================================================

def _log_results_to_csv(csv_path: str, row: dict) -> None:
    """
    Append one results row to a CSV. Creates the file with headers if it doesn't exist.
    Each sweep run appends one row — no manual copy-pasting from stdout.

    Args:
        csv_path: path to the CSV file (shared across all sweep runs)
        row: dict of column_name -> value for this run
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results logged to {csv_path}")


# =============================================================================
# MAIN GEAR UNLEARNING FUNCTION (contrastive loss + entanglement-score weighting)
# =============================================================================

def gear(ori_model, train_forget_loader, dt, dv, test_loader, device,
         poison_epoch=10, forget_class=0,
         to_forget=None, custom_forget=False, test_metadata=None, train_metadata=None, output_name=None,
         oculoplastics=False,
         retrain_model=None, train_remain_loader=None, remain_reg_param=1.0,
         selective_unlearning=False,
         # --------------------------------------------------------------
         # GEAR loss = beta * retain_loss + gamma_rep * contrastive_loss
         # beta:      remain_reg_param (retain CE weight)
         # gamma_rep: scales the combined contrastive/representation loss (CL+ES)
         # --------------------------------------------------------------
         feature_contrastive=False,
         feature_align_weight=0.0,
         retain_forget_weight=0.0,
         forget_forget_weight=0.0,
         gamma_rep=1.0,
         target_layer=9,
         results_csv=None,
         # ----------------------------------------------------------
         # ES (entanglement-score) weighting of rf_loss
         # ----------------------------------------------------------
         use_entanglement_weighting=False,
         centroid_refresh_interval=None,
         num_classes=None,
         cl_warmup_steps=0,
         data_name=None,
         ):
    """Trains an unlearn_model away from ori_model's weights by minimizing a
    retain cross-entropy loss plus a feature-space contrastive/entanglement
    loss — no forget-set cross-entropy term at all. Forgetting is achieved as
    a side effect of (a) never reinforcing the forget class and (b) actively
    repelling forget-sample representations away from retain-class
    representations in feature space (see retain_forget_loss/forget_forget_loss
    above), with the repulsion strength scaled per-sample by how "entangled"
    each forget sample currently is with the retain distribution when
    use_entanglement_weighting is enabled.

    data_name is only used to compute Retain Adjacent/Remote Accuracy (see
    class_hierarchy.py) - it's optional and has no effect on the unlearning
    algorithm itself; when omitted (or when the dataset has no known class
    hierarchy), those two metrics are reported as 'N/A'.
    """

    start = time.time()

    unlearn_model = copy.deepcopy(ori_model).to(device)

    # Frozen reference model — the pristine original weights, used to check
    # how far retain-set representations have drifted during unlearning.
    ref_model = copy.deepcopy(ori_model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    start_time = time.time()

    forget_data_gen = inf_generator(train_forget_loader)
    remain_data_gen = inf_generator(train_remain_loader)

    batches_per_epoch = len(train_forget_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.0001, momentum=0.9)

    num_iterations = poison_epoch * batches_per_epoch

    print(f'beta (remain_reg_param): {remain_reg_param} | gamma_rep: {gamma_rep}')
    print(f'retain_forget_weight: {retain_forget_weight} | forget_forget_weight: {forget_forget_weight}')
    print(f'target_layer: {target_layer}')

    # --- Entanglement-weighting setup ----------------------------------------
    if centroid_refresh_interval is None:
        centroid_refresh_interval = batches_per_epoch  # default: once per epoch

    # In multi-layer + entanglement mode, resolve the deepest layer dynamically
    # from the model rather than hardcoding RESNET_ALL_LAYERS[-1].
    # All other cases fall back to _get_primary_layer (existing behavior).
    _primary_layer = (
        _resolve_deepest_layer(unlearn_model, RESNET_ALL_LAYERS)
        if (use_entanglement_weighting and target_layer == 'all')
        else _get_primary_layer(target_layer)
    )
    _num_classes = num_classes        # may be None; inferred lazily on first use
    centroid_cache = None             # plain tensor buffer, refreshed every K steps

    if use_entanglement_weighting:
        print(f'use_entanglement_weighting: True | '
              f'centroid_refresh_interval: {centroid_refresh_interval} | '
              f'primary_layer: {_primary_layer}')
    # -------------------------------------------------------------------------

    # Per-component loss log for plotting
    loss_log = {
        "retain_loss": [],
        "contrastive_loss": [],
        "total_loss": [],
        "entanglement_score": [],   # mean e_i per step (0.0 when ES disabled)
    }

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    for itr in tqdm.tqdm(range(num_iterations)):

        x, y = forget_data_gen.__next__()
        x_rem, y_rem = remain_data_gen.__next__()

        x = x.to(device)
        y = y.to(device)
        if len(y.shape) > 1:
            y = y.squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)

        x_rem = x_rem.to(device)
        y_rem = y_rem.to(device)

        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        # --- CL warm-up ramp: scale gamma_rep from 0 → gamma_rep over
        # cl_warmup_steps from training start. No-op when cl_warmup_steps == 0.
        if cl_warmup_steps > 0 and itr < cl_warmup_steps:
            _cl_gamma_rep = gamma_rep * (itr / cl_warmup_steps)
        else:
            _cl_gamma_rep = gamma_rep

        # --- Retain loss (beta = remain_reg_param) -------------------------
        remain_logits = None  # initialised here so entanglement block can reuse it
        if remain_reg_param > 0:
            remain_logits = unlearn_model(x_rem)
            remain_loss = criterion(remain_logits, y_rem)
            remain_contrib = remain_reg_param * remain_loss
        else:
            remain_loss = torch.tensor(0.0, device=device)
            remain_contrib = torch.tensor(0.0, device=device)

        # --- Entanglement scores (computed before contrastive block) ----------
        # e_scores[i] = max_c cos(z_i^f, μ_c^r) for each forget sample i.
        # Used as weights in retain_forget_loss (CL component).
        # All feature extraction here is no-grad.
        e_scores = None           # [B_f] or None when disabled
        _ff_for_ent = None        # no-grad forget primary-layer features
        mean_ent_score = 0.0

        if use_entanglement_weighting:
            # Lazy num_classes inference (once per run)
            if _num_classes is None:
                if remain_logits is not None:
                    _num_classes = remain_logits.size(1)
                else:
                    with torch.no_grad():
                        _num_classes = unlearn_model(x_rem[:1]).size(1)

            # Refresh centroids on first step and every centroid_refresh_interval steps
            if centroid_cache is None or (itr > 0 and itr % centroid_refresh_interval == 0):
                centroid_cache = compute_retain_centroids(
                    unlearn_model, train_remain_loader, _primary_layer, _num_classes, device
                )
                # compute_retain_centroids restores training mode internally

            if centroid_cache is not None:
                if target_layer == 'all' and feature_contrastive:
                    # Multi-layer mode with contrastive: the multi-layer forward pass
                    # in the contrastive block already extracts all layer features.
                    # e_scores and _ff_for_ent will be derived from
                    # forget_feats_dict[_primary_layer] there — no separate pass needed.
                    pass
                else:
                    # Single-layer mode OR multi-layer without contrastive:
                    # extract primary-layer features now (existing behavior).
                    with torch.no_grad():
                        _ff_for_ent = prepare_features(
                            get_intermediate_features(unlearn_model, x, _primary_layer)
                        )  # [B_f, D]
                        e_scores = compute_entanglement_scores(_ff_for_ent, centroid_cache)
                        mean_ent_score = e_scores.mean().item()

        # --- Contrastive / representation loss (gamma_rep) -----------------
        contrastive_loss_val = torch.tensor(0.0, device=device)

        if feature_contrastive:
            if target_layer == 'all':
                layers = RESNET_ALL_LAYERS
                forget_feats_dict = get_intermediate_features_multilayer(unlearn_model, x, layers)
                retain_feats_dict = get_intermediate_features_multilayer(unlearn_model, x_rem, layers)
                with torch.no_grad():
                    retain_ref_feats_dict = get_intermediate_features_multilayer(ref_model, x_rem, layers)

                # Reuse deepest-layer features for entanglement scoring — avoids an
                # extra forward pass that _get_primary_layer used to require.
                # Only runs when entanglement weighting is enabled; single-layer path
                # is never reached by this branch.
                if use_entanglement_weighting and centroid_cache is not None:
                    with torch.no_grad():
                        _ff_for_ent = prepare_features(forget_feats_dict[_primary_layer].detach())
                        e_scores = compute_entanglement_scores(_ff_for_ent, centroid_cache)
                        mean_ent_score = e_scores.mean().item()

                align_loss = torch.tensor(0.0, device=device)
                rf_loss = torch.tensor(0.0, device=device)
                ff_loss = torch.tensor(0.0, device=device)
                for layer in layers:
                    f_f = prepare_features(forget_feats_dict[layer])
                    r_f = prepare_features(retain_feats_dict[layer])
                    r_ref_f = prepare_features(retain_ref_feats_dict[layer])
                    align_loss = align_loss + retain_alignment_loss(r_f, r_ref_f)
                    rf_loss = rf_loss + retain_forget_loss(r_f, f_f, entanglement_scores=e_scores)
                    ff_loss = ff_loss + forget_forget_loss(f_f)

                if itr % 100 == 0:
                    print(f"multi-layer contrastive over {layers}")
            else:
                forget_feats = get_intermediate_features(unlearn_model, x, target_layer)
                retain_feats = get_intermediate_features(unlearn_model, x_rem, target_layer)

                with torch.no_grad():
                    retain_ref_feats = get_intermediate_features(ref_model, x_rem, target_layer)

                forget_feats = prepare_features(forget_feats)
                retain_feats = prepare_features(retain_feats)
                retain_ref_feats = prepare_features(retain_ref_feats)

                align_loss = retain_alignment_loss(retain_feats, retain_ref_feats)
                rf_loss = retain_forget_loss(retain_feats, forget_feats, entanglement_scores=e_scores)
                ff_loss = forget_forget_loss(forget_feats)

                if itr % 100 == 0:
                    print("forget_feats shape:", forget_feats.shape)
                    print("retain_feats shape:", retain_feats.shape)
                    print("retain_ref_feats shape:", retain_ref_feats.shape)

            contrastive_loss_val = (
                feature_align_weight * align_loss +
                retain_forget_weight * rf_loss +
                forget_forget_weight * ff_loss
            )

        contrastive_contrib = _cl_gamma_rep * contrastive_loss_val

        # --- Total loss (GEAR: beta * retain + gamma_rep * contrastive) -------
        loss = remain_contrib + contrastive_contrib

        # --- Log components ---------------------------------------------------
        loss_log["retain_loss"].append(
            remain_loss.item() if isinstance(remain_loss, torch.Tensor) else 0.0
        )
        loss_log["contrastive_loss"].append(contrastive_loss_val.item())
        loss_log["total_loss"].append(loss.item())
        loss_log["entanglement_score"].append(mean_ent_score)

        if itr % 100 == 0:
            print(f"itr {itr:5d} | "
                  f"retain: {loss_log['retain_loss'][-1]:.4f} | "
                  f"contrastive: {contrastive_loss_val.item():.4f} | "
                  f"entanglement: {mean_ent_score:.4f} | "
                  f"total: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

    print('GEAR unlearning time:', (time.time() - start_time))
    gear_time = time.time() - start_time

    # Save loss curves
    _plot_loss_curves(loss_log, output_name)

    torch.save(unlearn_model, output_name + '.pth')

    # -------------------------------------------------------------------------
    # EVALUATION — build loaders
    # -------------------------------------------------------------------------
    if not custom_forget:
        if selective_unlearning:
            test_forget_loader, test_remain_loader = get_forget_loader(dv, forget_class)
        else:
            test_forget_loader, test_remain_loader = get_forget_loader(dv, forget_class)
            _, train_remain_loader = get_forget_loader(dt, forget_class)
    elif custom_forget and not oculoplastics:
        test_forget_loader, test_remain_loader = get_custom_forget_loader(dv, test_metadata, to_forget)
        _, train_remain_loader = get_custom_forget_loader(dt, train_metadata, to_forget)
    elif custom_forget and oculoplastics:
        test_forget_loader, test_remain_loader = get_custom_forget_loader_oculoplastics(dv, test_metadata)
        _, train_remain_loader = get_custom_forget_loader_oculoplastics(dt, train_metadata)
    else:
        # Unreachable: custom_forget/oculoplastics are booleans and the three
        # branches above already cover every combination. Kept explicit so
        # test_forget_loader/test_remain_loader are never used unassigned.
        raise ValueError("Unhandled custom_forget/oculoplastics combination")

    mode = ''

    # --- Unlearned model metrics -------------------------------------------
    _, test_acc = eval(model=unlearn_model, data_loader=test_loader, mode=mode,
                       print_perform=False, device=device, name='test set all class')
    _, forget_acc = eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode,
                         print_perform=False, device=device, name='test set forget class')
    _, remain_acc = eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode,
                         print_perform=False, device=device, name='test set remain class')

    # --- Retain Adjacent/Remote Accuracy (class-taxonomy-based - see
    # class_hierarchy.py; 'N/A' for any data_name without a known hierarchy,
    # including every custom_forget/oculoplastics clinical dataset) ---------
    adjacent_indices, remote_indices = class_hierarchy.get_adjacent_remote_split(
        data_name, forget_class, dv)
    retain_adjacent_acc, retain_remote_acc = class_hierarchy.compute_split_accuracy(
        unlearn_model, dv, adjacent_indices, remote_indices, device)

    retain_cos_sim = retain_intermediate_cosine_similarity(
        ref_model, unlearn_model, test_remain_loader, device, target_layer)
    retain_l1_diff = retain_intermediate_feature_l1_diff(
        ref_model, unlearn_model, test_remain_loader, device, target_layer)

    mia_mean, mia_std = compute_mia(unlearn_model, test_forget_loader, test_loader, device)
    lt_mia_result = compute_loss_threshold_mia(unlearn_model, train_forget_loader, test_forget_loader, device)
    lt_mia_auc = lt_mia_result['mia_auc']
    lt_mia_acc = lt_mia_result['mia_acc']

    print(f'intermediate layer l1 feature diff: {retain_l1_diff:.8f} at layer {target_layer}')
    print(f'intermediate layer cosine similarity: {retain_cos_sim:.8f} at layer {target_layer}')
    print(f'MIA score: {mia_mean:.4f} ± {mia_std:.4f}')
    print(f'MIA (loss-thr): acc={lt_mia_acc:.4f} auc={lt_mia_auc:.4f}')
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}'.format(test_acc, forget_acc, remain_acc))

    # --- Retrain baseline metrics (gold standard) --------------------------
    retrain_metrics = {}
    if retrain_model is not None:
        retrain_model.to(device)
        retrain_model.eval()

        _, rt_forget_acc = eval(model=retrain_model, data_loader=test_forget_loader, mode=mode,
                                print_perform=False, device=device, name='retrain forget')
        _, rt_remain_acc = eval(model=retrain_model, data_loader=test_remain_loader, mode=mode,
                                print_perform=False, device=device, name='retrain remain')
        rt_cos_sim = retain_intermediate_cosine_similarity(
            ref_model, retrain_model, test_remain_loader, device, target_layer)
        rt_l1_diff = retain_intermediate_feature_l1_diff(
            ref_model, retrain_model, test_remain_loader, device, target_layer)
        rt_mia_mean, rt_mia_std = compute_mia(retrain_model, test_forget_loader, test_loader, device)
        rt_lt_mia_result = compute_loss_threshold_mia(retrain_model, train_forget_loader, test_forget_loader, device)
        rt_lt_mia_auc = rt_lt_mia_result['mia_auc']
        rt_lt_mia_acc = rt_lt_mia_result['mia_acc']

        retrain_metrics = {
            "retrain_forget_acc": rt_forget_acc.item() if isinstance(rt_forget_acc, torch.Tensor) else rt_forget_acc,
            "retrain_remain_acc": rt_remain_acc.item() if isinstance(rt_remain_acc, torch.Tensor) else rt_remain_acc,
            "retrain_cos_sim": rt_cos_sim,
            "retrain_l1_diff": rt_l1_diff,
            "retrain_mia_mean": rt_mia_mean,
            "retrain_mia_std": rt_mia_std,
            "retrain_lt_mia_auc": rt_lt_mia_auc,
            "retrain_lt_mia_acc": rt_lt_mia_acc,
        }

        print(f'[RETRAIN BASELINE] '
              f'forget_acc: {rt_forget_acc:.2%} | '
              f'remain_acc: {rt_remain_acc:.2%} | '
              f'cos_sim: {rt_cos_sim:.4f} | '
              f'l1_diff: {rt_l1_diff:.4f} | '
              f'MIA (conf-clf): {rt_mia_mean:.4f} ± {rt_mia_std:.4f} | '
              f'MIA (loss-thr): acc={rt_lt_mia_acc:.4f} auc={rt_lt_mia_auc:.4f}')
    else:
        print('[RETRAIN BASELINE] No retrain model provided — skipping baseline evaluation.')

    # --- Log everything to CSV ---------------------------------------------
    if results_csv is not None:
        row = {
            # Hyperparameters
            "output_name": output_name,
            "beta": remain_reg_param,
            "gamma_rep": gamma_rep,
            "feature_align_weight": feature_align_weight,
            "retain_forget_weight": retain_forget_weight,
            "forget_forget_weight": forget_forget_weight,
            "remain_reg_param": remain_reg_param,
            "target_layer": target_layer,
            "poison_epoch": poison_epoch,
            "feature_contrastive": feature_contrastive,
            "use_entanglement_weighting": use_entanglement_weighting,
            "centroid_refresh_interval": centroid_refresh_interval,
            # Entanglement diagnostics (mean over all training steps)
            "mean_entanglement_score": (
                float(np.mean(loss_log["entanglement_score"]))
                if loss_log["entanglement_score"] else float('nan')
            ),
            # Unlearned model metrics
            "test_acc": test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc,
            "forget_acc": forget_acc.item() if isinstance(forget_acc, torch.Tensor) else forget_acc,
            "remain_acc": remain_acc.item() if isinstance(remain_acc, torch.Tensor) else remain_acc,
            "retain_adjacent_acc": retain_adjacent_acc,
            "retain_remote_acc": retain_remote_acc,
            "cos_sim": retain_cos_sim,
            "l1_drift": retain_l1_diff,
            "mia_mean": mia_mean,
            "mia_std": mia_std,
            "lt_mia_auc": lt_mia_auc,
            "lt_mia_acc": lt_mia_acc,
            "unlearning_time": gear_time,
        }
        row.update(retrain_metrics)
        _log_results_to_csv(results_csv, row)

    end = time.time()
    print('Time Consuming:', end - start, 'secs')

    unlearn_model.to(device)
    return unlearn_model, forget_acc, remain_acc, gear_time, test_acc, mia_mean, retain_adjacent_acc, retain_remote_acc
