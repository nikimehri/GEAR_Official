"""Cheng et al. unlearning baseline ("Machine Unlearning under Retain-Forget
Entanglement," 2026, arXiv:2603.26569,
https://github.com/Jingpu-Cheng/unlearning-entanglement).

The paper doesn't give its own method a public name/acronym (their own
reference code just labels it "ours" internally) - this module and its
`--method cheng_unlearn` key exist to cite the paper unambiguously. Not to
be confused with `baselines/baseline_main.py`'s unrelated sweep-mode `chen`/
`ravi` checkpoint baselines - those are a different paper's comparison
methods for the clinical datasets.

Reimplemented from the paper's description and the reference code's
structure - the reference repo has no LICENSE file, so this is a
reimplementation, not a port (same treatment as baselines/delete.py). Some
details (noted inline below) were reconstructed from a compressed research
pass rather than a byte-for-byte verified reference implementation - treat
hyperparameter defaults and the exact Stage 2 update rule as a documented,
reasonable-best-effort reading, not a guaranteed-exact match to the paper.

IMPORTANT compatibility constraint, different from every other baseline in
this repo: this method needs the Retain Adjacent/Remote split
(class_hierarchy.get_adjacent_remote_split) as an actual TRAINING input, not
just an evaluation-time metric - both stages train directly against the
retain-adjacent and retain-remote subsets. It is therefore only applicable
to datasets with a known class hierarchy (currently cifar100/tinyimagenet,
not cifar10) - there is nothing for it to constrain against otherwise.
cheng_unlearn does not attempt to run without a hierarchy; callers should
check class_hierarchy.get_adjacent_remote_split's return before calling in.

Algorithm (two stages, run sequentially):

Stage 1 (augmented-Lagrangian constrained forgetting, _stage1_lagrange):
maximizes (clipped) forget-set cross-entropy while a dual-ascent Lagrange
multiplier constrains mean cross-entropy on the retain-REMOTE split to stay
near its value under the original (pre-unlearning) model:
    L = gamma * L_f - clip(L_f, max=c) + lambda * g + (mu / 2) * g^2
    g = L_rem - L_rem(original_model)
    lambda <- lambda + mu * g          (dual ascent, applied each step)

Stage 2 (W2-regularized gradient-projected fine-tuning, _stage2_wpgd):
continues training against a forget loss blending cross-entropy with a
Wasserstein-2 penalty (approximated via sorted per-sample losses) against a
frozen reference model (the end of Stage 1); the retain-ADJACENT gradient is
computed each step and projected onto the orthogonal complement of
{forget gradient, retain-remote gradient} before being added to the update -
so it can help the closest retain classes without undoing the forgetting or
the remote-retain protection Stage 1 established. The exact combination
rule (project-then-sum vs. adjacent-only) wasn't independently verified
against the reference code; this implementation sums all three (forget,
remote, projected-adjacent) into a single SGD step per iteration - momentum
is standard SGD momentum applied to that combined update, not a separate
mechanism specific to the adjacent gradient.

Fully architecture-agnostic (works via model(x)/model.parameters() and
plain gradient computation, no hooked intermediate layer) - unlike
cfk/euk/coun it needs no per-architecture branching to run on
AllCNN/ResNet/ViT.
"""
import copy

import torch
from torch import nn


def _mean_loss_over_loader(model, loader, device):
    """Mean cross-entropy loss over a full loader pass, no grad - used for
    Stage 1's retain-remote anchor (the original model's own mean loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1:
                y = y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            total_loss += criterion(model(x), y).item()
            total_count += y.size(0)
    return total_loss / max(total_count, 1)


def _wasserstein2_proxy(losses_a, losses_b):
    """1D Wasserstein-2 distance proxy between two per-sample loss
    distributions, via sorted values (the paper's own approximation) -
    truncates to the shorter of the two if batch sizes differ."""
    n = min(losses_a.size(0), losses_b.size(0))
    sorted_a, _ = torch.sort(losses_a[:n])
    sorted_b, _ = torch.sort(losses_b[:n])
    return torch.mean((sorted_a - sorted_b) ** 2)


def _next_batch(gen_state, device):
    """Advances an _inf_generator iterator, normalizing label shape the
    same way every other baseline in this repo does."""
    x, y = next(gen_state)
    x, y = x.to(device), y.to(device)
    if y.dim() > 1:
        y = y.squeeze()
    if y.dim() == 0:
        y = y.unsqueeze(0)
    return x, y


def _inf_generator(loader):
    """Endlessly cycles a DataLoader (same idea as gear.py's inf_generator,
    duplicated here to keep this baseline importable without pulling in
    gear.py's much larger module)."""
    while True:
        for batch in loader:
            yield batch


def _stage1_lagrange(model, forget_loader, remote_loader, original_model, device,
                      num_epochs=1, lr=2.5e-6, mu=10.0, gamma=1.0, c=10.0):
    """Stage 1: augmented-Lagrangian constrained forgetting. Returns a new
    model (a deep copy of `model`, trained); `model` itself is untouched."""
    model = copy.deepcopy(model).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    l_rem_anchor = _mean_loss_over_loader(original_model, remote_loader, device)

    remote_gen = _inf_generator(remote_loader)
    lam = 0.0  # Lagrange multiplier, dual-ascent updated every step

    for _epoch in range(num_epochs):
        for x_f, y_f in forget_loader:
            x_f, y_f = x_f.to(device), y_f.to(device)
            if y_f.dim() > 1:
                y_f = y_f.squeeze()
            if y_f.dim() == 0:
                y_f = y_f.unsqueeze(0)
            x_r, y_r = _next_batch(remote_gen, device)

            optimizer.zero_grad()

            l_f = criterion(model(x_f), y_f)
            l_f_clipped = torch.clamp(l_f, max=c)

            l_rem = criterion(model(x_r), y_r)
            g = l_rem - l_rem_anchor

            loss = gamma * l_f - l_f_clipped + lam * g + (mu / 2) * (g ** 2)
            loss.backward()
            optimizer.step()

            lam = lam + mu * g.item()

    model.eval()
    return model


def _stage2_wpgd(model, forget_loader, remote_loader, adjacent_loader, ref_model, device,
                  num_epochs=6, lr=2e-5, momentum=0.9, alpha=0.5):
    """Stage 2: W2-regularized, adjacent-retain gradient-projected
    fine-tuning. Trains `model` in place (already a fresh copy from Stage 1)
    and returns it."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    remote_gen = _inf_generator(remote_loader)
    adjacent_gen = _inf_generator(adjacent_loader)

    for _epoch in range(num_epochs):
        for x_f, y_f in forget_loader:
            x_f, y_f = x_f.to(device), y_f.to(device)
            if y_f.dim() > 1:
                y_f = y_f.squeeze()
            if y_f.dim() == 0:
                y_f = y_f.unsqueeze(0)
            x_r, y_r = _next_batch(remote_gen, device)
            x_a, y_a = _next_batch(adjacent_gen, device)

            optimizer.zero_grad()

            # Forget loss: CE blended with a W2 penalty against the frozen
            # Stage-1 reference's per-sample loss distribution on the same batch.
            per_sample_ce = criterion_none(model(x_f), y_f)
            with torch.no_grad():
                ref_per_sample_ce = criterion_none(ref_model(x_f), y_f)
            w2 = _wasserstein2_proxy(per_sample_ce, ref_per_sample_ce)
            l_f_tilde = (1 - alpha) * per_sample_ce.mean() + alpha * w2
            grad_f = torch.autograd.grad(l_f_tilde, model.parameters(), allow_unused=True)

            l_rem = criterion(model(x_r), y_r)
            grad_rem = torch.autograd.grad(l_rem, model.parameters(), allow_unused=True)

            l_adj = criterion(model(x_a), y_a)
            grad_adj = torch.autograd.grad(l_adj, model.parameters(), allow_unused=True)

            # Project the adjacent-retain gradient onto the orthogonal
            # complement of {grad_f, grad_rem}, then combine all three into
            # one update - see module docstring for why.
            with torch.no_grad():
                for p, gf, gr, ga in zip(model.parameters(), grad_f, grad_rem, grad_adj):
                    combined = torch.zeros_like(p)
                    if gf is not None:
                        combined = combined + gf
                    if gr is not None:
                        combined = combined + gr
                    if ga is not None:
                        g = ga.clone()
                        for basis in (gf, gr):
                            if basis is not None:
                                denom = torch.sum(basis * basis)
                                if denom > 1e-12:
                                    g = g - (torch.sum(g * basis) / denom) * basis
                        combined = combined + g
                    p.grad = combined

            optimizer.step()

    model.eval()
    return model


def cheng_unlearn(model, forget_loader, adjacent_loader, remote_loader, device,
                   stage1_epochs=1, stage1_lr=2.5e-6, mu=10.0, gamma=1.0, c=10.0,
                   stage2_epochs=6, stage2_lr=2e-5, momentum=0.9, alpha=0.5):
    """Returns a new, unlearned copy of model (model itself is not modified).
    Runs Stage 1 (augmented-Lagrangian constrained forgetting against the
    retain-remote split) then Stage 2 (W2-regularized fine-tuning with the
    retain-adjacent gradient projected orthogonal to forget/remote).

    adjacent_loader/remote_loader must come from
    class_hierarchy.get_adjacent_remote_split - callers should check that
    function's return isn't (None, None) before calling in here (see module
    docstring: this baseline only applies to datasets with a known class
    hierarchy)."""
    original_model = copy.deepcopy(model).to(device)
    original_model.eval()
    for p in original_model.parameters():
        p.requires_grad_(False)

    stage1_model = _stage1_lagrange(
        model, forget_loader, remote_loader, original_model, device,
        num_epochs=stage1_epochs, lr=stage1_lr, mu=mu, gamma=gamma, c=c,
    )

    # Stage 2's frozen reference is the end of Stage 1 (the paper's own
    # design auto-derives this when a separate reference isn't supplied).
    stage1_reference = copy.deepcopy(stage1_model).to(device)
    stage1_reference.eval()
    for p in stage1_reference.parameters():
        p.requires_grad_(False)

    final_model = _stage2_wpgd(
        stage1_model, forget_loader, remote_loader, adjacent_loader, stage1_reference, device,
        num_epochs=stage2_epochs, lr=stage2_lr, momentum=momentum, alpha=alpha,
    )

    final_model.eval()
    return final_model
