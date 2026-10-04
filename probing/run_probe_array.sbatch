#!/bin/bash
# Linear-probe evaluation as a SLURM job-array task (one checkpoint per task).
#
# GENERIC PLACEHOLDER - every #SBATCH value and module-load line below is a
# TODO. This repo has no existing SLURM infrastructure to pattern-match
# against (no #SBATCH references anywhere else in the codebase), so none of
# these are filled in with real partition/account/module names - fill them
# in for whichever cluster you actually run this on before submitting.
#
# Usage:
#   1. Edit CHECKPOINTS/ARCHES/DATASETS/FORGET_CLASSES/SEEDS below (or
#      generate checkpoints.txt some other way - one line per task, see the
#      format note above the array below).
#   2. sbatch probing/run_probe_array.sbatch

#SBATCH --job-name=linear_probe
#SBATCH --array=0-TODO                 # TODO: one index per line in checkpoints.txt
#SBATCH --partition=TODO               # TODO: your cluster's GPU partition name
#SBATCH --account=TODO                 # TODO: your allocation/account name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=probing/logs/%x_%A_%a.out
#SBATCH --error=probing/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p probing/logs probing/results probing/feature_cache probing/recipes

# TODO: module loads / environment activation for your cluster, e.g.:
#   module load python/3.11 cuda/12.1
#   source .venv/bin/activate
# or, if using conda:
#   module load miniconda3
#   conda activate gear

# checkpoints.txt format: one task per line, whitespace-separated fields:
#   <checkpoint_path> <arch> <dataset> <forget_class> <seed> <method_name>
#
# Example line:
#   model_checkpoints/baseline_models/allcnn_cifar10_delete_forget0_seed42.pth allcnn cifar10 0 42 delete
#
# Generate this file however fits your workflow (a small Python/shell loop
# over your checkpoint directory is usually easiest) before submitting.
CHECKPOINTS_FILE="probing/checkpoints.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHECKPOINTS_FILE")
read -r CKPT ARCH DATASET FORGET_CLASS SEED METHOD <<< "$LINE"

RECIPE_JSON="probing/recipes/${ARCH}_${DATASET}_recipe.json"
OUTPUT_JSON="probing/results/${METHOD}_${ARCH}_${DATASET}_forget${FORGET_CLASS}_seed${SEED}.json"

# The recipe must already exist (run --select_recipe once per (arch, dataset)
# against the Original checkpoint, outside the array, before submitting this
# job) - every array task reads the same frozen weight_decay, never re-tunes.
if [ ! -f "$RECIPE_JSON" ]; then
    echo "ERROR: $RECIPE_JSON not found. Run linear_probe.py --select_recipe" \
         "against this config's Original checkpoint first." >&2
    exit 1
fi

python probing/linear_probe.py \
    --checkpoint "$CKPT" \
    --arch "$ARCH" \
    --dataset "$DATASET" \
    --forget_class "$FORGET_CLASS" \
    --seed "$SEED" \
    --method "$METHOD" \
    --recipe_json "$RECIPE_JSON" \
    --output_json "$OUTPUT_JSON" \
    --cache_dir probing/feature_cache \
    --ncc
