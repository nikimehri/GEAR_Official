# Legacy PSG (boundary-shrinkage) launch scripts

These scripts invoke `main.py` with `--sgld`, `--closest_points`, `--gamma`,
`--lamda`, `--use_logits`, and/or `--logit_preprocess` — flags that only
meant something under `boundary_unlearning.py`'s boundary-shrinkage
algorithm (the original PSG paper's method). That algorithm was replaced by
`gear.py`'s contrastive/entanglement-weighted method, and those flags were
removed from `params.py` accordingly, so these scripts will now fail with
an "unrecognized arguments" error if run as-is.

They're kept here, archived rather than deleted or silently left broken, as
a historical record of the hyperparameters and experiment configurations
used for the original paper's CIFAR-10/FashionMNIST/clinical-imaging
results. If you need to reproduce one of these runs under GEAR, the
`--do_unlearning`/`--run_sota`/`--specific_settings`/`--forget_class`/
`--custom_unlearn`-style flags still work the same way — you'd just drop
the PSG-specific flags above and add GEAR's contrastive-loss/entanglement-
score flags instead (see the main README).

Scripts still at `SHELL_SCRIPTS/`'s top level (`tsne.sh`,
`cifar/master_ensemble.sh`, `cifar/master_retrain_cifar_all_class.sh`)
don't reference any of the removed flags and still work unchanged.
