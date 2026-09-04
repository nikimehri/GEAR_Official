# GEAR: Contrastive/Entanglement-Weighted Machine Unlearning

GEAR removes the influence of specific training samples (a class, a device, a
diagnosis) from a trained model without full retraining. It builds on top of
an earlier bilevel-optimization approach — "Targeted Unlearning Using
Perturbed Sign Gradient Methods" (Nahass et al., 2025, cited below) — but
replaces that paper's boundary-shrinkage algorithm with a feature-space
contrastive method: it minimizes a retain-set cross-entropy loss plus a
contrastive loss that pushes forget-sample representations away from
retain-class representations, optionally weighting that push per-sample by
an **entanglement score** (how similar a forget sample's current
representation already is to some retain class). There is no forget-set
cross-entropy term at all — forgetting is a side effect of never
reinforcing the forget class while actively repelling its representations
from the retain distribution.

## Architecture

**Main pipeline** (repo root):
- `main.py` — CLI entry point: builds the dataset/train-val-test split, trains
  or loads an original+retrain model pair, and (with `--do_unlearning`) runs
  `gear.py`'s unlearning.
- `gear.py` — the unlearning algorithm itself: the contrastive/entanglement
  loss functions, the training loop (with GEAR-dynamic/GEAR-cached retain-
  centroid modes, see `--centroid_mode` in the
  [CLI reference](#cli-reference-mainpy) below), and all evaluation metrics
  (accuracy, Retain Adjacent/Remote Accuracy, AIN, representation drift,
  membership-inference attacks).
- `models.py` — `AllCNN` (CIFAR-10/FashionMNIST/SVHN/MedMNIST), `CustomResNet`
  (ResNet-50, for CIFAR-100/TinyImageNet and the clinical datasets), and
  `ViT` (pretrained ViT-Base/32 @ 224x224, for CIFAR-100/TinyImageNet; the
  clinical pipeline reuses the same class with a different patch/image-size
  config), all exposing a `forward_with_features`/`get_embedding` hook API
  that `gear.py`'s contrastive losses depend on.
- `class_hierarchy.py` — the CIFAR-100 (official) and TinyImageNet
  (WordNet-hypernym-derived approximation) class-superclass tables used to
  compute Retain Adjacent/Remote Accuracy, shared by `gear.py` and every
  baseline rather than duplicated.
- `ain_metric.py` — the Anamnesis Index (AIN) implementation ("relearn
  time" fine-tuning + ratio against a gold-standard retrain model), also
  shared by `gear.py` and the baselines.
- `make_dataloaders.py` — dataset loading (including TinyImageNet) and
  forget/remain split construction, for both class-based forgetting and
  metadata-attribute-based forgetting (clinical device/diagnosis/exam-year
  attributes).
- `trainer.py` — model construction, the training loop, and checkpoint
  loading/saving.
- `params.py`, `utils.py` — CLI argument parsing and small shared helpers.
- `embeddings.py`, `ensemble.py` — analysis/visualization utilities (t-SNE
  plots, an experimental classifier-splicing side-experiment).
- `compute_initial_es.py`, `evaluate_retrain.py` — standalone diagnostic
  scripts (see [Diagnostics](#diagnostics) below).
- `scripts/prepare_tinyimagenet.py`, `scripts/build_tinyimagenet_hierarchy.py`
  — one-time TinyImageNet setup scripts (see
  [Dataset Preparation](#dataset-preparation) below).

**Baselines** (`baselines/`): independent implementations of every
comparison method, dispatched through `baselines/baseline_main.py` — see
[Running the baselines](#running-the-baselines). `baselines/thirdparty/` is
vendored third-party distillation-loss code (RepDistiller) used by SCRUB and
a couple of the layer-freezing baselines.

**`SHELL_SCRIPTS/`**: launch scripts used for hyperparameter tuning and
experimentation. Scripts specific to the original boundary-shrinkage method
(which used different CLI flags) are archived under `SHELL_SCRIPTS/legacy_psg/`
with their own README explaining why.

## Setup

```bash
pip install -r requirements.txt
```

Checkpoint paths in `baselines/path_dicts.py` and `baselines/baseline_main.py`
are relative to a `MODEL_CHECKPOINT_ROOT` environment variable (defaults to
`./model_checkpoints`) — set it if your checkpoints live somewhere else:

```bash
export MODEL_CHECKPOINT_ROOT=/path/to/your/checkpoints
```

Note the clinical imaging datasets aren't distributed with this repo (see
[Data and Model Availability](#data-and-model-availability)) — the
open-source datasets (CIFAR-10/100, FashionMNIST, SVHN, MedMNIST) download
automatically via `torchvision`/`medmnist` on first use. TinyImageNet needs
one manual step first — see [Dataset Preparation](#dataset-preparation)
below. `nltk` is listed in `requirements.txt` but is a one-time/dev
dependency only — it's needed to *regenerate*
`class_hierarchy.py`'s TinyImageNet mapping, not to run experiments.

## Dataset Preparation

**TinyImageNet** isn't a built-in `torchvision.datasets` class and needs a
one-time download + reorganization step before `--data_name tinyimagenet`
will work:

```bash
python scripts/prepare_tinyimagenet.py --dataset_dir ./data
```

This downloads `tiny-imagenet-200.zip` (if not already present), extracts
it, and reorganizes its `val/` split into per-class subdirectories
(`train/` already ships that way) so it can be loaded via `ImageFolder`
the same way both splits are.

TinyImageNet has no official class-superclass table (unlike CIFAR-100), so
Retain Adjacent/Remote Accuracy for it uses an approximation built from
WordNet hypernym relationships, generated once via:

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet')"
python scripts/build_tinyimagenet_hierarchy.py --dataset_dir ./data
```

The resulting table is already committed in `class_hierarchy.py`
(`TINYIMAGENET_SUPERCLASS_MAPPING`) — you only need to re-run this script if
you want to regenerate it (e.g. with a different grouping granularity).

## Quick Start

**1. Train an original model and its retrain (gold-standard, remain-only)
counterpart:**

```bash
python main.py --forget_class 0 \
               --data_name fashionmnist \
               --model_name AllCNN \
               --lr 0.01 \
               --epoch 15 \
               --batch_size 64 \
               --gpu_id 0 \
               --train
```

**2. Run GEAR unlearning** against an existing original/retrain checkpoint
pair. `--specific_settings` evaluates against the held-out validation split;
`--run_sota` evaluates against the test set (both log to the CSV named by
`--name`, plus a shared `{name}_sweep_results.csv` with full per-run detail):

```bash
python main.py --forget_class 0 \
               --data_name cifar100 \
               --model_name resnet50 \
               --do_unlearning \
               --specific_settings \
               --feature_contrastive \
               --retain_forget_weight 2.0 \
               --forget_forget_weight 3.0 \
               --remain_reg 1.0 \
               --use_entanglement_weighting \
               --target_layer layer4 \
               --name 'unlearn_cifar100_class0' \
               --original_model 'model_checkpoints/.../original_model.pth' \
               --retrain_model 'model_checkpoints/.../retrain_model.pth'
```

**3. Selective/metadata-based unlearning** (forget samples matching a
clinical attribute rather than a whole class):

```bash
python main.py --custom_unlearn \
               --to_forget 'Cirrus 800 FA' \
               --data_name fundus_3_class \
               --do_unlearning ...
```

This mode requires the clinical metadata CSVs described in
[Data and Model Availability](#data-and-model-availability) — it won't run
without them.

## CLI Reference (`main.py`)

**Dataset / model**
| Flag | Meaning |
|---|---|
| `--data_name` | `cifar10`, `cifar100`, `tinyimagenet`, `fashionmnist`, `svhn`, `medmnist`, or a clinical dataset name |
| `--model_name` | `AllCNN`, `resnet`, `resnet50`, or `vit`. `cifar10`/`fashionmnist` require `AllCNN`; `cifar100`/`tinyimagenet` require `resnet`/`resnet50`/`vit`. `vit` resizes inputs to 224x224 with ImageNet normalization automatically (see `models.ViT`) |
| `--dataset_dir`, `--checkpoint_dir` | where data downloads to / checkpoints save to |
| `--val_fraction` | fraction of the training set held out for validation (default 0.1) |
| `--seed` | RNG seed, also seeds the train/val split |

**Mode** — pick one training/loading mode and (optionally) one unlearning mode:
| Flag | Meaning |
|---|---|
| `--train` | train an original + retrain model pair from scratch |
| `--retrain_only` | load an existing `--original_model`, train only the retrain model |
| *(neither)* | load both `--original_model` and `--retrain_model` from disk |
| `--do_unlearning` | required for either unlearning mode below |
| `--run_sota` | run GEAR, evaluate against the test set |
| `--specific_settings` | run GEAR, evaluate against the validation set |

**What to forget**
| Flag | Meaning |
|---|---|
| `--forget_class` | class index to forget (class-based mode) |
| `--custom_unlearn`, `--to_forget` | forget samples matching a metadata attribute instead of a class |
| `--oculoplastics` | use the oculoplastics dataset's metadata-threshold forgetting mode |
| `--selective_unlearn`, `--percent_to_forget` | forget only a percentage of the target class/attribute, not all of it |

**GEAR contrastive loss (CL) and entanglement score (ES)**
| Flag | Meaning |
|---|---|
| `--feature_contrastive` | enable the feature-space contrastive loss terms (off by default = plain retain-CE-only unlearning) |
| `--feature_align_weight` | weight on keeping retain features close to the frozen original model |
| `--retain_forget_weight` | weight on pushing retain and forget features apart |
| `--forget_forget_weight` | weight on pulling forget samples together with each other |
| `--gamma_rep` | overall scale on the combined contrastive loss |
| `--remain_reg` | weight on the plain retain cross-entropy loss |
| `--target_layer` | which layer to compute features at: `9` for AllCNN, `layer4` for ResNet-50, `all` for all four ResNet stages |
| `--use_entanglement_weighting` | weight the retain-forget push per-sample by each forget sample's entanglement score |
| `--centroid_refresh_interval` | training steps between retain-centroid recomputation when `--centroid_mode dynamic` (default: once per epoch) |
| `--centroid_mode` | `dynamic` (default) recomputes retain centroids periodically as the model's representations shift during unlearning; `cached` computes them once and freezes them for the run ("GEAR-dynamic" vs. "GEAR-cached"). Only meaningful alongside `--use_entanglement_weighting` |
| `--cl_warmup_steps` | linearly ramp the contrastive loss in over this many steps at the start of training |
| `--poison_epoch` | number of passes over the forget set during unlearning |

**Evaluation metrics**
| Flag | Meaning |
|---|---|
| *(none needed)* | Retain Adjacent/Remote Accuracy is always computed and reported (as `'N/A'` for any `--data_name` without a known class hierarchy — currently only `cifar100`/`tinyimagenet` have one; see `class_hierarchy.py`) |
| `--compute_ain` | compute the Anamnesis Index (AIN). Off by default — unlike the other metrics this involves actually fine-tuning a copy of the model ("relearn time"), not just an extra evaluation pass. Reported as `'N/A'` when off or when no `--retrain_model` was provided |
| `--ain_error_range` | AIN relearning target: fraction below the original model's forget-set accuracy considered "recovered" (default 0.05) |
| `--ain_lr` | learning rate for AIN's relearning-phase SGD optimizer (default 0.1) |
| `--ain_max_epochs` | max epochs of relearning before AIN reports non-convergence as `inf` (default 10) |
| `--ain_eval_interval` | mini-batch steps between AIN relearning-accuracy checks (default 50) |

**Analysis modes**
| Flag | Meaning |
|---|---|
| `--tsne_embeddings`, `--unlearn_model`, `--embeddings_name` | generate a 3-panel t-SNE comparison (original/retrain/unlearned) |
| `--ensemble`, `--good_forget`, `--good_remain` | experimental classifier-splicing side-experiment (not part of the paper's reported results) |

## Running the Baselines

Every baseline dispatches through `baselines/baseline_main.py` in
single-experiment mode (pass `--original_model` to bypass the hardcoded
sweep-mode experiment table):

```bash
python baselines/baseline_main.py \
    --data_name cifar100 --model_name resnet50 \
    --original_model model_checkpoints/.../original_model.pth \
    --retrain_model model_checkpoints/.../retrain_model.pth \
    --forget_class 0 \
    --method finetune,neggrad,cfk,euk,scrub,delete,ssd,coun \
    --name my_baseline_run
```

`--method` accepts a comma-separated subset of:
- `finetune` — continue training on the retain set only
- `neggrad` — minimize retain loss while maximizing (gradient-ascending) forget loss
- `cfk` — freeze everything except the last block, fine-tune it on the retain set
- `euk` — like `cfk`, but resets the last block's weights before fine-tuning it
- `scrub` — knowledge-distillation baseline (alternating maximize/minimize passes against a frozen teacher); pass `--feature_contrastive`/`--use_entanglement_weighting`/`--retain_forget_weight`/etc. (same meaning as `main.py`'s) to run SCRUB with GEAR's contrastive/entanglement regularizer for a head-to-head comparison under matching settings, or `--scrub_epochs` to control training length
- `delete` — DELETE (Decoupled Distillation to Erase, CVPR 2025): trains a copy of the model on the forget set only, distilling toward the frozen original model's own predictions with each sample's true-label logit masked out before softmax. No retain-set loss term. `--delete_epochs`/`--delete_lr`/`--delete_disable_bn` tune it. Reimplemented from the paper's description — the [reference repo](https://github.com/shaaaaron/DELETE) ships with no LICENSE file, so this is a clean reimplementation, not a code port
- `ssd` — SSD (Selective Synaptic Dampening, AAAI 2024): no training loop at all — computes per-parameter Fisher information on the forget set and on the full original trainset, then dampens (in place) any parameter disproportionately important to the forget set. `--ssd_dampening_constant`/`--ssd_selection_weighting` tune it (selection_weighting defaults to 5 for ViT, 10 otherwise, matching the reference's own architecture-aware default). Adapted from the [reference repo](https://github.com/if-loops/selective-synaptic-dampening) (MIT licensed)
- `coun` — CoUn (retain-only, self-supervised contrastive baseline; see `baselines/coun.py`'s module docstring). `--coun_epochs`/`--coun_lr`/`--coun_lambda_scale`/`--coun_temp` tune it — fixed here, not swept (see the standalone CLI below for the hyperparameter sweep)

Every baseline also reports Retain Adjacent/Remote Accuracy (`'N/A'` unless
`--data_name` is `cifar100`/`tinyimagenet`) automatically, and AIN when
`--compute_ain` is passed (plus `--ain_error_range`/`--ain_lr`/
`--ain_max_epochs`/`--ain_eval_interval` to tune it — same meaning and
defaults as `main.py`'s flags of the same name) — both computed once per
`all_readouts()` call, covering every method in `--method` with no other
flags needed.

Results are written to `{name}_{data_name}_results.json`.

**CoUn's standalone CLI**: for the `lambda_scale`/`temp` hyperparameter
sweep specifically (too expensive to run inline alongside every other
baseline in a `--method` batch), `baselines/coun.py` still runs on its own,
and — like every other baseline — now works across any of the 6 supported
configs via `--data_name`/`--model_name` (previously CIFAR-100/ResNet-50 only):

```bash
python baselines/coun.py \
    --data_name cifar100 --model_name resnet50 \
    --forget_class 0 --num_epochs 50 \
    --checkpoint model_checkpoints/.../original_model.pth \
    --retrain_checkpoint model_checkpoints/.../retrain_model.pth
```

By default this first sweeps `lambda_scale`/`temp` on a held-out seed, then
evaluates across seeds 45-50 and writes `coun_results.json`. Pass
`--skip_sweep --lambda_scale <v> --temp <v>` to skip the sweep.
`--retrain_checkpoint` is optional — CoUn also reports Retain Adjacent/Remote
Accuracy automatically (`'N/A'` unless `--data_name` is
`cifar100`/`tinyimagenet`), and AIN when a retrain checkpoint is given
(`'N/A'` otherwise).

### Baseline / configuration compatibility

Every baseline listed above (`finetune`, `neggrad`, `cfk`, `euk`, `scrub`,
`delete`, `ssd`, `coun`) works across all 6 supported model/dataset
configurations:

| | CIFAR-10/AllCNN | CIFAR-100/AllCNN | CIFAR-100/ResNet50 | TinyImageNet/ResNet50 | CIFAR-100/ViT | TinyImageNet/ViT |
|---|---|---|---|---|---|---|
| finetune / neggrad | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| cfk / euk | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| scrub | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| delete | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ssd | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| coun | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`finetune`/`neggrad`/`delete`/`ssd` are architecture-agnostic by
construction (they only ever touch `model.parameters()`/`model(x)`).
`cfk`/`euk`/`coun` each resolve a "last representational block" per
architecture (`--target_layer`-style convention: `features[9]` for AllCNN,
`resnet_base.layer4` for ResNet, `vit.blocks[-1]` for ViT). `scrub`'s core
distillation loop is architecture-agnostic; its optional
`--feature_contrastive`/`--use_entanglement_weighting` CL+ES mode needs
`--target_layer` set correctly for whichever architecture is in play, same
as `main.py`'s GEAR runs. `chen`/`ravi`/`eval_orig` (sweep-mode only) are
pre-baked comparison checkpoints specific to the clinical datasets, not
general unlearning methods — out of scope for this matrix.

## Diagnostics

- `compute_initial_es.py` — measures how entangled the forget class already
  is in the *original* (pre-unlearning) model, across several seeds, to
  characterize the baseline entanglement level before any unlearning.
- `evaluate_retrain.py` — sanity-checks a retrain checkpoint's accuracy on
  the remain portion of the test set, independent of running `main.py`.

## Known Issues

- Retain Adjacent/Remote Accuracy and AIN are only computed for
  `--data_name cifar100`/`tinyimagenet` (Retain Adjacent/Remote Accuracy) or
  when a `--retrain_model`/`--retrain_checkpoint` is available (AIN) —
  every other case reports `'N/A'` rather than an error.
- TinyImageNet has no official class-superclass table (unlike CIFAR-100), so
  its Retain Adjacent/Remote Accuracy grouping (`class_hierarchy.py`'s
  `TINYIMAGENET_SUPERCLASS_MAPPING`) is a WordNet-hypernym-based
  approximation, not an authoritative reproduction of anything the reference
  paper published — see `scripts/build_tinyimagenet_hierarchy.py`'s
  docstring for the exact algorithm.
- `--run_sota` doesn't report Retain Adjacent/Remote Accuracy or AIN (only
  `--specific_settings` does) — both were added to match what the
  validation-set evaluation path already tracked.
- `mnist` is not a supported `--data_name`/`--model_name` pairing (no working
  model architecture exists for it in this repo).
- Metadata-based (`--custom_unlearn`) unlearning on the clinical datasets
  requires clinical metadata CSVs not distributed with this repo — see
  below.

## Data and Model Availability

[Original/retrain checkpoints for the open-source datasets are available for download here](https://drive.google.com/drive/folders/1fBa1BhOXKdjBCWEM00OjZAeRsm3pUpjx).

The clinical imaging datasets (fundus, oculoplastics, MRI, ultrasound, OCT)
and their metadata are not distributed with this repo due to data-sharing
restrictions.

## Cite Us

This repo builds on:

```bibtex
@misc{nahass2025targetedunlearningusingperturbed,
      title={Targeted Unlearning Using Perturbed Sign Gradient Methods With Applications On Medical Images},
      author={George R. Nahass and Zhu Wang and Homa Rashidisabet and Won Hwa Kim and Sasha Hubschman and Jeffrey C. Peterson and Chad A. Purnell and Pete Setabutr and Ann Q. Tran and Darvin Yi and Sathya N. Ravi},
      year={2025},
      eprint={2505.21872},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.21872},
}
```

Two evaluation metrics reported alongside GEAR and every baseline are also
implementations of ideas from other papers (see `class_hierarchy.py` and
`ain_metric.py`'s module docstrings for the exact definitions used and how
they differ from each paper's own reference code):

```bibtex
@misc{cheng2026retainforgetentanglement,
      title={Machine Unlearning under Retain-Forget Entanglement},
      eprint={2603.26569},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2603.26569},
}

@misc{chundawat2023zeroshot,
      title={Zero-Shot Machine Unlearning},
      author={Vikram S Chundawat and Ayush K Tarun and Murari Mandal and Mohan Kankanhalli},
      year={2023},
      eprint={2201.05629},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2201.05629},
}
```

## Contact

For questions, contact gnahas2@uic.edu and sathya@uic.edu
