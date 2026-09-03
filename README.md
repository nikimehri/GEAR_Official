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
  loss functions, the training loop, and all evaluation metrics (accuracy,
  representation drift, membership-inference attacks).
- `models.py` — `AllCNN` (CIFAR-10/FashionMNIST/SVHN/MedMNIST) and
  `CustomResNet` (ResNet-50, for CIFAR-100 and the clinical datasets), both
  exposing a `forward_with_features` hook API that `gear.py`'s contrastive
  losses depend on.
- `make_dataloaders.py` — dataset loading and forget/remain split
  construction, for both class-based forgetting and metadata-attribute-based
  forgetting (clinical device/diagnosis/exam-year attributes).
- `trainer.py` — model construction, the training loop, and checkpoint
  loading/saving.
- `params.py`, `utils.py` — CLI argument parsing and small shared helpers.
- `embeddings.py`, `ensemble.py` — analysis/visualization utilities (t-SNE
  plots, an experimental classifier-splicing side-experiment).
- `compute_initial_es.py`, `evaluate_retrain.py` — standalone diagnostic
  scripts (see [Diagnostics](#diagnostics) below).

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
automatically via `torchvision`/`medmnist` on first use.

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
| `--data_name` | `cifar10`, `cifar100`, `fashionmnist`, `svhn`, `medmnist`, or a clinical dataset name |
| `--model_name` | `AllCNN`, `resnet`, `resnet50`, or `vit`. `cifar10`/`fashionmnist` require `AllCNN`; `cifar100` requires `resnet`/`resnet50` |
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
| `--centroid_refresh_interval` | training steps between retain-centroid recomputation (default: once per epoch) |
| `--cl_warmup_steps` | linearly ramp the contrastive loss in over this many steps at the start of training |
| `--poison_epoch` | number of passes over the forget set during unlearning |

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
    --method finetune,neggrad,cfk,euk,scrub \
    --name my_baseline_run
```

`--method` accepts a comma-separated subset of:
- `finetune` — continue training on the retain set only
- `neggrad` — minimize retain loss while maximizing (gradient-ascending) forget loss
- `cfk` — freeze everything except the last block, fine-tune it on the retain set
- `euk` — like `cfk`, but resets the last block's weights before fine-tuning it
- `scrub` — knowledge-distillation baseline (alternating maximize/minimize passes against a frozen teacher); pass `--feature_contrastive`/`--use_entanglement_weighting`/`--retain_forget_weight`/etc. (same meaning as `main.py`'s) to run SCRUB with GEAR's contrastive/entanglement regularizer for a head-to-head comparison under matching settings, or `--scrub_epochs` to control training length

Results are written to `{name}_{data_name}_results.json`.

**CoUn** (a retain-only, self-supervised contrastive baseline — see
`baselines/coun.py`'s module docstring) runs as its own script, since it's
CIFAR-100/ResNet-50-specific and doesn't share the rest of the pipeline's
data loading:

```bash
python baselines/coun.py --forget_class 0 --num_epochs 50 \
    --checkpoint model_checkpoints/.../original_model.pth
```

By default this first sweeps `lambda_scale`/`temp` on a held-out seed, then
evaluates across seeds 45-50 and writes `coun_results.json`. Pass
`--skip_sweep --lambda_scale <v> --temp <v>` to skip the sweep.

## Diagnostics

- `compute_initial_es.py` — measures how entangled the forget class already
  is in the *original* (pre-unlearning) model, across several seeds, to
  characterize the baseline entanglement level before any unlearning.
- `evaluate_retrain.py` — sanity-checks a retrain checkpoint's accuracy on
  the remain portion of the test set, independent of running `main.py`.

## Known Issues

- The `--specific_settings`/`--run_sota` results CSV includes `Retain Remote
  Acc` and `Retain Adjacent Acc` columns that are currently always `N/A` —
  these metrics aren't computed anywhere yet; a definition is pending.
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

## Contact

For questions, contact gnahas2@uic.edu and sathya@uic.edu
