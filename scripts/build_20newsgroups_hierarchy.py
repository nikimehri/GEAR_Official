"""One-time generation script for 20 Newsgroups' class hierarchy.

Unlike TinyImageNet (scripts/build_tinyimagenet_hierarchy.py's WordNet-based
approximation), 20 Newsgroups has a well-known, standard 6-supercategory
grouping by topic (computers/forsale/politics/recreation/religion/science),
widely used in NLP literature discussing this dataset - this script just
verifies that grouping against sklearn's actual, live `target_names`
ordering (rather than hand-typing a fine-label-index array and hoping the
ordering matches), and prints the resulting Python literal to paste into
class_hierarchy.py.

Usage:
    pip install scikit-learn
    python scripts/build_20newsgroups_hierarchy.py
"""
from collections import Counter

from sklearn.datasets import fetch_20newsgroups

# Newsgroups whose supercategory isn't determined by their dotted prefix
# alone (comp.*/rec.*/sci.* all map to one group each via PREFIX_GROUP
# below; these don't share a prefix with their topical siblings).
GROUP_OVERRIDES = {
    'alt.atheism': 'religion',
    'soc.religion.christian': 'religion',
    'talk.religion.misc': 'religion',
    'talk.politics.guns': 'politics',
    'talk.politics.mideast': 'politics',
    'talk.politics.misc': 'politics',
    'misc.forsale': 'forsale',
}
PREFIX_GROUP = {'comp': 'computers', 'rec': 'recreation', 'sci': 'science'}


def group_for(name):
    if name in GROUP_OVERRIDES:
        return GROUP_OVERRIDES[name]
    return PREFIX_GROUP[name.split('.')[0]]


if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train')
    names = data.target_names
    if len(names) != 20:
        raise ValueError(f"Expected 20 newsgroups, sklearn returned {len(names)}")

    assignment = {name: group_for(name) for name in names}
    group_names = sorted(set(assignment.values()))
    group_to_idx = {g: i for i, g in enumerate(group_names)}
    mapping = [group_to_idx[assignment[name]] for name in names]

    print(f'# {len(group_names)} supercategories over {len(names)} classes.')
    print('# Supercategories (name -> member newsgroups):')
    members = {g: [] for g in group_names}
    for name in names:
        members[assignment[name]].append(name)
    for g in group_names:
        print(f'#   {g}: {members[g]}')
    print(f'# Sizes: {dict(sorted(Counter(mapping).items()))}')

    print()
    print('TWENTYNEWSGROUPS_SUPERCLASS_MAPPING = [')
    print('    ' + ', '.join(str(x) for x in mapping) + ',')
    print(']')
