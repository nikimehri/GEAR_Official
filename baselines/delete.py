"""DELETE baseline (Decoupled Distillation to Erase, CVPR 2025 Highlight).

Clean reimplementation from the algorithm described in the paper and its
reference repo (https://github.com/shaaaaron/DELETE.git) - not a port of
the reference code, which ships with no LICENSE file, so this cites the
idea rather than reusing their source.

Core idea: the frozen original model's own prediction on each forget-set
sample is used as a distillation target, but with that sample's own
true-label logit masked to a very large negative number before softmax -
producing a target distribution with near-zero mass on the class being
forgotten. A copy of the model (the "student") is trained to match this
masked target via KL divergence, using only the forget set - there is no
retain-set loss term at all; retain-set performance is only ever evaluated
here, never trained against.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn


def _set_bn_eval(model):
    """Puts every BatchNorm layer into eval mode (frozen running stats)
    while everything else stays in train mode - used when disable_bn=True,
    since there's no retain-set signal here to keep BN running stats sane
    during the short forget-only fine-tune."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def delete_unlearn(model, train_forget_loader, device, unlearn_epoch=20, unlearn_rate=1e-4, disable_bn=False):
    """Returns a new, unlearned copy of model (model itself is untouched).

    Masks each forget sample's own label index (not a single hardcoded
    forget_class), so this also behaves correctly for metadata-based
    (--custom_unlearn) forgetting, not just class-based forgetting."""
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(model).to(device)
    student.train()
    if disable_bn:
        _set_bn_eval(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=unlearn_rate)

    for _epoch in range(unlearn_epoch):
        for x, y in train_forget_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1:
                y = y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)

            with torch.no_grad():
                teacher_logits = teacher(x)
                masked_logits = teacher_logits.clone()
                masked_logits.scatter_(1, y.unsqueeze(1), -1e10)
                teacher_target = F.softmax(masked_logits, dim=1)

            student_log_probs = F.log_softmax(student(x), dim=1)
            loss = F.kl_div(student_log_probs, teacher_target, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    return student
