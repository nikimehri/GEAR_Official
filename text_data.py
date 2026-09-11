"""20 Newsgroups dataset loading, matching the packed-tensor convention
text_models.TextTransformer expects (see that module's docstring). Kept in
its own module, imported lazily (only from inside the '20newsgroups'
branches of get_dataset), for the same reason text_models.py is separate:
isolates the new scikit-learn-fetch/`transformers`-tokenizer dependency so
it can never break an existing vision experiment that doesn't need it.
"""
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """Tokenizes texts once at construction time and stores them as packed
    [2, seq_len] tensors (input_ids, attention_mask stacked at dim 0) - see
    text_models.TextTransformer's docstring for why this packing exists.

    Exposes .targets/.imgs/.class_to_idx as duck-typed stand-ins for the
    attributes trainer.py's train_save_model and utils.py's set_num_classes
    already read off of ImageFolder/CIFAR-style datasets (dataset.imgs[i][1]
    for per-sample labels, dataset.class_to_idx for the idx_to_class
    fallback) - this is what lets those two functions work on this dataset
    completely unmodified, with zero new branches added to either file."""

    def __init__(self, texts, labels, tokenizer, class_names, max_length=256):
        encoded = tokenizer(
            list(texts), padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        # Stack once at construction, not per-__getitem__ call.
        self.packed = torch.stack([encoded['input_ids'], encoded['attention_mask']], dim=1)  # [N, 2, seq_len]
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.targets = self.labels.tolist()
        # ImageFolder-shaped stand-in: a list of (path_placeholder, label)
        # tuples - trainer.py only ever reads index [1] (the label) off this.
        self.imgs = [(None, label) for label in self.targets]
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Plain int, not a 0-dim tensor - matches CIFAR/ImageFolder's own
        # __getitem__ convention (their labels come from plain Python lists,
        # not a tensor), which class_hierarchy.py's set-membership checks
        # (`label in adjacent_classes`) rely on: a 0-dim tensor doesn't hash
        # the same as the plain int it numerically equals, so `tensor(15) in
        # {15, 19}` is silently False even though the value matches. DataLoader
        # batching is unaffected either way - collate re-tensorizes labels
        # from a list of ints or a list of 0-dim tensors identically.
        return self.packed[idx], int(self.labels[idx])


def get_20newsgroups_datasets(path='./data', model_name='distilbert-base-uncased', max_length=256):
    """Returns (trainset, testset, dataset) - same 3-tuple convention as
    make_dataloaders.get_dataset - for 20 Newsgroups. `dataset` (the
    untransformed reference copy used for class-count/label lookups
    elsewhere) is the same object as testset here, since tokenization always
    happens up front (there's no separate "untransformed" variant to fall
    back to) - mirroring how a couple of make_dataloaders.py's own branches
    (svhn, fashionmnist) already reuse testset as `dataset` too.

    `remove=('headers', 'footers', 'quotes')` is scikit-learn's own
    documented recommendation for this dataset - without it, a classifier
    can trivially "cheat" off near-unique header metadata (e.g. an
    X-Newsreader field) instead of learning from the actual message content."""
    from sklearn.datasets import fetch_20newsgroups
    from transformers import AutoTokenizer

    train_raw = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), data_home=path)
    test_raw = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), data_home=path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trainset = TextClassificationDataset(train_raw.data, train_raw.target, tokenizer,
                                         train_raw.target_names, max_length=max_length)
    testset = TextClassificationDataset(test_raw.data, test_raw.target, tokenizer,
                                        test_raw.target_names, max_length=max_length)

    return trainset, testset, testset
