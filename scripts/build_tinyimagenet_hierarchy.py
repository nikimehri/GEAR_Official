"""One-time, offline generation script for TinyImageNet's approximate class
hierarchy (there is no official TinyImageNet superclass table, unlike
CIFAR-100). Groups TinyImageNet's 200 WordNet-synset classes into coarse
semantic groups using nltk's WordNet corpus, and prints a Python literal
(`TINYIMAGENET_SUPERCLASS_MAPPING`) to paste into class_hierarchy.py.

This script is NOT part of the runtime path - it's run once, by hand, to
regenerate the mapping if TinyImageNet's wnids.txt or the grouping algorithm
changes. Runtime code (class_hierarchy.py) only ever reads the static table
this script produces; it never imports nltk or touches the network.

Algorithm: for each class's WordNet synset, walk its primary hypernym path
(root -> synset) and assign it to the *most specific* ancestor whose total
descendant count among the 200 classes is >= MIN_GROUP_SIZE - this favors
tight, semantically meaningful groups (e.g. "carnivore" for dogs/cats/bears)
over the very lopsided partition a fixed tree-depth cutoff produces (a couple
of giant branches dominate WordNet's upper levels). When a synset has more
than one hypernym path (WordNet permits multiple inheritance), the path is
chosen deterministically (lexicographically smallest sequence of ancestor
names) so the result doesn't depend on Python's hash-randomization seed.

Usage:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet')"
    python scripts/build_tinyimagenet_hierarchy.py --dataset_dir ./data
"""
import argparse
import os
from collections import Counter

MIN_GROUP_SIZE = 8


def load_wnids(dataset_dir):
    """Reads the 200 class WordNet IDs from wnids.txt, sorted alphabetically
    - this matches the order torchvision.datasets.ImageFolder assigns class
    indices in (it sorts subdirectory names, and TinyImageNet's per-class
    subdirectories are named exactly by wnid), so index i in the mapping
    this script produces lines up with fine-label index i at runtime."""
    path = os.path.join(dataset_dir, 'tiny-imagenet-200', 'wnids.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_tinyimagenet.py first."
        )
    with open(path, 'r') as f:
        wnids = sorted(line.strip() for line in f if line.strip())
    if len(wnids) != 200:
        raise ValueError(f"Expected 200 wnids, found {len(wnids)} in {path}")
    return wnids


def primary_hypernym_path(synset):
    """Picks one root->synset hypernym path deterministically. A synset can
    have multiple such paths (multiple inheritance) - hypernym_paths()'s own
    ordering isn't guaranteed stable across processes, so this breaks ties by
    sorting candidate paths by their sequence of ancestor names."""
    paths = synset.hypernym_paths()
    paths.sort(key=lambda path: [s.name() for s in path])
    return paths[0]


def build_mapping(wnids):
    """Returns a length-200 list mapping each wnid's position (in the sorted
    wnids order) to a 0-indexed coarse-group id."""
    from nltk.corpus import wordnet as wn

    synsets = {wnid: wn.synset_from_pos_and_offset('n', int(wnid[1:])) for wnid in wnids}
    paths = {wnid: primary_hypernym_path(s) for wnid, s in synsets.items()}

    # How many of the 200 classes descend from each ancestor synset.
    coverage = Counter()
    for path in paths.values():
        for ancestor in path:
            coverage[ancestor.name()] += 1

    assignment = {}
    for wnid, path in paths.items():
        # Walk from the immediate parent up toward the root, keeping the
        # first (most specific) ancestor with enough coverage. Falls back to
        # the root itself if nothing else qualifies.
        chosen = path[0].name()
        for ancestor in reversed(path[:-1]):
            if coverage[ancestor.name()] >= MIN_GROUP_SIZE:
                chosen = ancestor.name()
                break
        assignment[wnid] = chosen

    group_names = sorted(set(assignment.values()))
    group_to_idx = {name: i for i, name in enumerate(group_names)}
    mapping = [group_to_idx[assignment[wnid]] for wnid in wnids]

    return mapping, group_names, assignment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TinyImageNet\'s WordNet-hypernym-based class hierarchy.')
    parser.add_argument('--dataset_dir', type=str, default='./data',
                        help='directory tiny-imagenet-200/ (with wnids.txt) lives under')
    args = parser.parse_args()

    wnid_list = load_wnids(args.dataset_dir)
    class_mapping, superclass_names, wnid_to_superclass = build_mapping(wnid_list)

    print(f'# {len(superclass_names)} superclasses over {len(wnid_list)} classes.')
    print('# Superclasses (WordNet synset name -> member wnids):')
    from collections import defaultdict
    members = defaultdict(list)
    for wnid in wnid_list:
        members[wnid_to_superclass[wnid]].append(wnid)
    for name in superclass_names:
        print(f'#   {name}: {members[name]}')

    print()
    print('TINYIMAGENET_SUPERCLASS_MAPPING = [')
    for i in range(0, len(class_mapping), 10):
        print('    ' + ', '.join(str(x) for x in class_mapping[i:i + 10]) + ',')
    print(']')
