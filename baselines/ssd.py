"""SSD baseline (Selective Synaptic Dampening, AAAI 2024).

Adapted from https://github.com/if-loops/selective-synaptic-dampening
(MIT licensed). A pure gradient/weight-space method - no training loop, no
retain-set access, and no retrain/gold model needed. Computes per-parameter
Fisher information (mean squared gradient of cross-entropy loss, backprop-
accumulated over a dataloader) twice: once on the forget set, once on the
full original training set (forget+retain combined, not just the retain
split). Parameters where forget-set importance exceeds
selection_weighting x original-importance are dampened in place, scaled
down proportionally to how much more important they were to the forget set
than to the whole dataset.
"""
import copy

import torch
from torch import nn


class ParameterPerturber:
    """Computes per-parameter Fisher-information "importance" for a model
    and applies the SSD dampening update to it in place."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def calc_importance(self, dataloader):
        """Mean squared-gradient importance per named parameter, averaged
        over every sample in dataloader (not just every batch, so a
        smaller/larger final batch doesn't skew the average)."""
        importances = {name: torch.zeros_like(p, device=self.device)
                       for name, p in self.model.named_parameters()}
        self.model.eval()
        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            if y.dim() > 1:
                y = y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)

            self.model.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()

            batch_size = x.size(0)
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    importances[name] += (p.grad.detach() ** 2) * batch_size
            total_samples += batch_size

        for name in importances:
            importances[name] /= max(total_samples, 1)

        self.model.zero_grad()
        return importances

    def modify_weight(self, original_importance, forget_importance,
                       dampening_constant=1.0, selection_weighting=10.0,
                       lower_bound=1.0, exponent=1.0):
        """In-place SSD dampening update over every named parameter: where
        a parameter's forget-set importance exceeds selection_weighting x
        its whole-dataset importance, scale it down by
        min(((original_importance * dampening_constant) / forget_importance)
        ** exponent, lower_bound) - lower_bound=1 means the factor only
        ever shrinks a weight, never grows it."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                oimp = original_importance[name]
                fimp = forget_importance[name]

                # Parameters with zero forget-set importance are never
                # selected below (fimp > threshold is False when fimp==0,
                # since the threshold is >= 0) - this guard just avoids a
                # literal 0/0 while computing the (unused, in that case)
                # dampening factor at those positions.
                safe_fimp = torch.where(fimp > 0, fimp, torch.ones_like(fimp))
                selected = fimp > (selection_weighting * oimp)

                dampening_factor = torch.clamp(
                    (oimp * dampening_constant / safe_fimp) ** exponent,
                    max=lower_bound,
                )
                update = torch.where(selected, dampening_factor, torch.ones_like(dampening_factor))
                p.mul_(update)


def ssd_unlearn(model, forget_loader, full_train_loader, device,
                 dampening_constant=1.0, selection_weighting=None, model_name=None):
    """Returns a new, dampened copy of model (model itself is untouched).

    selection_weighting=None resolves to 5.0 for transformer architectures
    (ViT, DistilBERT), 10.0 otherwise - matching the reference
    implementation's own architecture-aware default (their
    model_size_scaler halves alpha for transformers)."""
    if selection_weighting is None:
        selection_weighting = 5.0 if model_name in ('vit', 'distilbert') else 10.0

    unlearned_model = copy.deepcopy(model).to(device)
    perturber = ParameterPerturber(unlearned_model, device)

    forget_importance = perturber.calc_importance(forget_loader)
    original_importance = perturber.calc_importance(full_train_loader)

    perturber.modify_weight(original_importance, forget_importance,
                            dampening_constant=dampening_constant,
                            selection_weighting=selection_weighting)

    unlearned_model.eval()
    return unlearned_model
