"""CU baseline (Contrastive Unlearning, Lee et al. 2024,
https://arxiv.org/abs/2401.10458).

Not to be confused with baselines/coun.py's "CoUn" (a different paper,
Khalil et al. 2025) - both are kept as separate baselines.

Reimplementation from the paper's Algorithm 1 / Equations 5-7 (no reference
code repository is linked from the paper). Core idea: a "reversed"
InfoNCE-style contrastive loss on raw penultimate embeddings (no projection
head) - for each forget-set sample x_i, same-class retain samples ("P_z",
called positives by the paper's own labeling convention) are pushed away,
and different-class retain samples ("N_z", negatives) are pulled close:

    L_UL = (-1/|N_z(x_i)|) * sum_{z_a in N_z(x_i)} log[
               exp(z_i . z_a / tau) / sum_{z_p in P_z(x_i)} exp(z_i . z_p / tau)
           ]

For whole-class forgetting, P_z(x_i) is empty (no retain samples share the
forgotten class) - the paper's Equation 6 substitutes a constant |N_z(x_i)|
for the denominator in that case (this reimplementation does the same,
avoiding an empty log-sum-exp instead of literally dividing by zero).

Combined with a plain retain-set cross-entropy loss:

    L = lambda_UL * L_UL + lambda_CE * L_CE(F(X^r), Y^r)

Per forget batch, the retain batch is resampled and the combined loss/step
repeated omega times (paper: omega <= 4). No frozen reference/teacher model
or retrain/gold model is needed by the algorithm itself.

The paper doesn't state numeric values for tau/lambda_UL/lambda_CE/learning
rate/optimizer/epochs - the defaults below are this reimplementation's own
reasonable choices, documented as such rather than claimed as the paper's.

Because the loss operates on each model's own get_embedding(x) (the raw
pooled pre-classifier representation) rather than a hooked intermediate
layer, this works unmodified across AllCNN/CustomResNet/ViT/DistilBERT - no
per-architecture branching needed, unlike cfk/euk/coun.
"""
import copy
import sys
import os

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gear import inf_generator
from baseline_utils import eval as eval_accuracy


def _cu_loss(forget_embeddings, forget_labels, retain_embeddings, retain_labels, temperature):
    """Per-forget-sample reversed-InfoNCE loss (paper Eq. 5/6), averaged over
    the forget batch. Forget samples with no different-class retain sample in
    this particular retain batch (N_z empty) are skipped - the retain batch
    would need to be all one class for that to happen, which practically
    only matters for very small batch sizes."""
    losses = []
    for i in range(forget_embeddings.size(0)):
        z_i = forget_embeddings[i]
        y_i = forget_labels[i]

        same_mask = (retain_labels == y_i)
        diff_mask = ~same_mask

        z_neg = retain_embeddings[diff_mask]
        if z_neg.size(0) == 0:
            continue

        sim_neg = torch.matmul(z_neg, z_i) / temperature

        z_pos = retain_embeddings[same_mask]
        if z_pos.size(0) > 0:
            # Eq. 5 (sample unlearning): normalize against same-class retain similarities.
            log_denom = torch.logsumexp(torch.matmul(z_pos, z_i) / temperature, dim=0)
        else:
            # Eq. 6 (class unlearning): no same-class retain samples exist -
            # constant "damping" denominator instead of an empty log-sum-exp.
            log_denom = torch.log(torch.tensor(float(z_neg.size(0)), device=z_i.device))

        losses.append(-(sim_neg - log_denom).mean())

    if not losses:
        return torch.zeros((), device=forget_embeddings.device, requires_grad=True)
    return torch.stack(losses).mean()


def cu_unlearn(model, train_forget_loader, train_remain_loader, device,
                num_classes, lambda_ul=1.0, lambda_ce=1.0, temperature=0.1,
                omega=4, lr=0.01, max_epochs=10, eval_forget_loader=None):
    """Returns a new, unlearned copy of model (model itself is not modified).

    eval_forget_loader, when given, enables the paper's own early-termination
    check (Algorithm 1): after each epoch, stop once forget-set accuracy
    drops to or below 1/num_classes (the class-unlearning termination
    criterion). When omitted, runs the full max_epochs unconditionally -
    matching every other baseline's simpler fixed-epoch convention."""
    model = copy.deepcopy(model).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    retain_gen = inf_generator(train_remain_loader)

    for _epoch in range(max_epochs):
        for x_u, y_u in train_forget_loader:
            x_u, y_u = x_u.to(device), y_u.to(device)
            if y_u.dim() > 1:
                y_u = y_u.squeeze()
            if y_u.dim() == 0:
                y_u = y_u.unsqueeze(0)

            for _ in range(omega):
                x_r, y_r = next(retain_gen)
                x_r, y_r = x_r.to(device), y_r.to(device)
                if y_r.dim() > 1:
                    y_r = y_r.squeeze()
                if y_r.dim() == 0:
                    y_r = y_r.unsqueeze(0)

                optimizer.zero_grad()

                z_u = F.normalize(model.get_embedding(x_u), dim=-1)
                z_r = F.normalize(model.get_embedding(x_r), dim=-1)

                loss_ul = _cu_loss(z_u, y_u, z_r, y_r, temperature)
                loss_ce = nn.CrossEntropyLoss()(model(x_r), y_r)
                loss = lambda_ul * loss_ul + lambda_ce * loss_ce

                loss.backward()
                optimizer.step()

        if eval_forget_loader is not None:
            model.eval()
            _, forget_acc = eval_accuracy(model=model, data_loader=eval_forget_loader, device=device)
            model.train()
            if forget_acc <= (1.0 / num_classes):
                break

    model.eval()
    return model
