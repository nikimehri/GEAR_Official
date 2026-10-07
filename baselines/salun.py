"""SalUn baseline (Saliency Unlearning, Fan et al., ICLR 2024 Spotlight,
"SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in
Both Image Classification and Generation", https://github.com/OPTML-Group/Unlearn-Saliency).

Adapted from the reference repo (MIT licensed) - specifically
Classification/generate_mask.py's save_gradient_ratio and
Classification/unlearn/RL.py's RL function. Two stages, both reimplemented
here from the real source (not from memory of the paper alone):

Stage 1 (compute_saliency_mask): accumulate the gradient of the forget-set
cross-entropy loss over every forget-set batch (summed, not averaged),
take the absolute value per-parameter, then select the globally top
mask_ratio fraction of parameters by magnitude (ranked across every
parameter in the model together, not per-tensor) - a binary mask marking
which weights are "salient" to the forget set and therefore allowed to
change during unlearning.

Stage 2 (salun_unlearn, the reference's "RL" = Random Labeling): relabels
every forget-set image with a uniform-random label (which may occasionally
coincide with the true label - the reference implementation doesn't
exclude that case either) and fine-tunes on relabeled-forget + true-labeled
retain samples jointly, but only ever applies gradient updates to the
salient (masked-in) weights - masked-out gradients are zeroed before every
optimizer step, and masked-out weights are forcibly restored to their
exact pre-unlearning value after every step (including resetting any SGD
momentum buffer on those weights) so residual momentum can't leak updates
into supposedly-frozen parameters across steps. This restore step is a real
correctness detail in the reference code, not a simplification - without
it, momentum alone would still slowly drift masked-out weights away from
their saliency-protected value.

One simplification from the reference, verified not to change the result:
the reference accumulates the gradient of the NEGATED forget loss (gradient
ascent direction) before taking the absolute value; since abs(-x) == abs(x),
this implementation accumulates the plain (non-negated) gradient instead,
which is mathematically identical and clearer. A second simplification: the
reference branches its training-loop mixing strategy by dataset name
(CIFAR-10/SVHN run two separate per-epoch passes; CIFAR-100/TinyImageNet
concatenate-and-shuffle forget+retain together) - this implementation always
uses the simpler, unified concatenate-and-pair-per-batch strategy (one
forget batch paired with one retain batch per step, via inf_generator to
cycle the usually-larger retain loader), matching the cifar100/tinyimagenet
branch's approach uniformly rather than special-casing by dataset.

mask_ratio defaults to 0.5 (a commonly-reported CIFAR configuration in the
paper) - not independently re-verified against every per-table value the
paper reports, consistent with how this project documents similarly
reconstructed hyperparameter choices elsewhere (e.g. cheng_unlearn.py).
"""
import copy

import torch
import torch.nn as nn

from gear import inf_generator


def compute_saliency_mask(model, forget_loader, device, mask_ratio=0.5):
    """Returns {param_name: binary mask tensor}, selecting the globally top
    mask_ratio fraction of parameters by |gradient of forget-set CE loss|.
    Operates on a deep copy of model - never modifies the caller's model or
    its gradients."""
    model = copy.deepcopy(model).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    gradients = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters()}

    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        if y.dim() > 1:
            y = y.squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)

        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] += param.grad.detach()

    for name in gradients:
        gradients[name] = gradients[name].abs()

    all_grads = torch.cat([g.flatten() for g in gradients.values()])
    k = int(len(all_grads) * mask_ratio)
    if k <= 0:
        return {name: torch.zeros_like(g) for name, g in gradients.items()}
    if k >= len(all_grads):
        return {name: torch.ones_like(g) for name, g in gradients.items()}

    threshold = torch.topk(all_grads, k, largest=True).values.min()
    return {name: (g >= threshold).float() for name, g in gradients.items()}


def _apply_mask_to_grads(model, mask):
    """Zeroes the gradient of every masked-out (non-salient) parameter,
    in place, before the optimizer step consumes it."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad.mul_(mask[name])


def _restore_masked_params(model, mask, theta0, optimizer):
    """Forces every masked-out parameter back to its exact pre-unlearning
    value (theta0) after the optimizer step, and zeroes any SGD momentum
    buffer on it - without this, residual momentum alone (even with a
    zeroed current-step gradient) would still slowly drift masked-out
    weights away from the value saliency was supposed to protect."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            inv_mask = 1.0 - mask[name]
            if torch.count_nonzero(inv_mask) == 0:
                continue
            param.data.mul_(mask[name]).add_(theta0[name] * inv_mask)
            state = optimizer.state.get(param, None)
            if state is not None and state.get('momentum_buffer') is not None:
                state['momentum_buffer'].mul_(mask[name])


def salun_unlearn(model, train_forget_loader, train_remain_loader, device, num_classes,
                  mask_ratio=0.5, unlearn_epochs=10, lr=0.01, momentum=0.9, weight_decay=5e-4,
                  lr_decay_epochs=(5, 8)):
    """Returns a new, unlearned copy of model (model itself is untouched).

    lr_decay_epochs mirrors the reference's decreasing_lr milestone list for
    a MultiStepLR schedule (gamma fixed at 0.1, matching the reference's own
    hardcoded value)."""
    mask = compute_saliency_mask(model, train_forget_loader, device, mask_ratio=mask_ratio)

    student = copy.deepcopy(model).to(device)
    with torch.no_grad():
        theta0 = {name: p.detach().clone() for name, p in student.named_parameters() if name in mask}

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(lr_decay_epochs), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    retain_gen = inf_generator(train_remain_loader)
    student.train()

    for _epoch in range(unlearn_epochs):
        for x_f, _y_f in train_forget_loader:
            x_f = x_f.to(device)
            random_labels = torch.randint(0, num_classes, (x_f.size(0),), device=device)

            x_r, y_r = next(retain_gen)
            x_r, y_r = x_r.to(device), y_r.to(device)
            if y_r.dim() > 1:
                y_r = y_r.squeeze()
            if y_r.dim() == 0:
                y_r = y_r.unsqueeze(0)

            x = torch.cat([x_f, x_r], dim=0)
            y = torch.cat([random_labels, y_r], dim=0)

            optimizer.zero_grad()
            loss = criterion(student(x), y)
            loss.backward()

            _apply_mask_to_grads(student, mask)
            optimizer.step()
            _restore_masked_params(student, mask, theta0, optimizer)

        scheduler.step()

    student.eval()
    return student
