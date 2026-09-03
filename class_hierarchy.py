"""Shared class-hierarchy definitions and Retain Adjacent/Remote Accuracy
helpers, imported by gear.py, baselines/baseline_main.py, and
baselines/coun.py alike - one implementation instead of three duplicated
copies.

Retain Adjacent/Remote Accuracy (Cheng et al., "Machine Unlearning under
Retain-Forget Entanglement," https://arxiv.org/abs/2603.26569) partitions the
retain set relative to a forgotten class using a fixed, a-priori class
taxonomy: "adjacent" retain samples belong to a class sharing the forgotten
class's coarse superclass; "remote" retain samples belong to every other
superclass. Both are reported as ordinary micro-averaged top-1 accuracy -
this is not a distance/embedding-based metric, confirmed against the paper's
own evaluation code:
https://github.com/Jingpu-Cheng/unlearning-entanglement/blob/main/src/evaluation.py
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# CIFAR-100's official 20-superclass grouping (fine label index -> coarse
# label index), in torchvision.datasets.CIFAR100's fine-label ordering
# (alphabetical by class name - the order the dataset's own meta file uses).
CIFAR100_SUPERCLASS_MAPPING = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def _get_superclass_mapping(data_name):
    """Returns the fine-label -> coarse-label list for a dataset, or None if
    no hierarchy is known for it - signals "not applicable" to callers,
    which should report 'N/A' rather than erroring."""
    if data_name == 'cifar100':
        return CIFAR100_SUPERCLASS_MAPPING
    return None


def get_adjacent_remote_split(data_name, forget_class, dataset):
    """Given a fine-label forget_class, returns (adjacent_indices,
    remote_indices): indices into dataset of retain-set samples (forget_class
    samples themselves are excluded from both) whose class shares the
    forgotten class's superclass (adjacent) vs. doesn't (remote). Returns
    (None, None) for datasets with no known class hierarchy."""
    mapping = _get_superclass_mapping(data_name)
    if mapping is None:
        return None, None

    forget_superclass = mapping[forget_class]
    adjacent_classes = {
        fine for fine, coarse in enumerate(mapping)
        if coarse == forget_superclass and fine != forget_class
    }

    adjacent_indices = []
    remote_indices = []
    for i, (_, label) in enumerate(dataset):
        if label == forget_class:
            continue
        if label in adjacent_classes:
            adjacent_indices.append(i)
        else:
            remote_indices.append(i)

    return adjacent_indices, remote_indices


def compute_split_accuracy(model, dataset, indices_adjacent, indices_remote, device, batch_size=64):
    """Ordinary micro-averaged top-1 accuracy (correct/total pooled across
    batches) over the adjacent and remote index sets, matching the reference
    paper's evaluate_on_three_datasets aggregation. Each of the two returned
    values is 'N/A' when its index list is None (no known hierarchy) or
    empty (no matching samples in this particular split)."""

    def _accuracy_over(indices):
        if not indices:
            return 'N/A'
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=SubsetRandomSampler(indices))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                if target.dim() > 1:
                    target = target.squeeze()
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 'N/A'

    acc_adjacent = _accuracy_over(indices_adjacent) if indices_adjacent is not None else 'N/A'
    acc_remote = _accuracy_over(indices_remote) if indices_remote is not None else 'N/A'
    return acc_adjacent, acc_remote
