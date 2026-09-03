#!/usr/bin/env python3
"""Compute per-sample retain-proximity weights (e_i) on the pretrained model, per seed.

This is NOT Zhao et al.'s dataset-level Entanglement Score.
This is the per-sample max-cosine-similarity quantity that the unlearning code uses
as a weight on the retain-forget repulsion loss.
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import random_split

from make_dataloaders import dataloader_engine, get_dataset
from trainer import load_checkpoint
from utils import set_num_classes
from gear import (
    compute_retain_centroids,
    compute_entanglement_scores,
    prepare_features,
    get_intermediate_features,
)


def main():
    """For each --seeds value: rebuilds the train/val split with that seed,
    computes retain-class centroids from the (untouched) checkpoint, then
    computes every forget sample's entanglement score against those
    centroids. Prints a per-seed summary table and writes the full
    per-sample results to initial_es_{data_name}.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True,
                        help='Pretrained original model checkpoint.')
    parser.add_argument('--data_name', default='cifar100')
    parser.add_argument('--dataset_dir', default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--target_layer', default='layer4')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    # dataloader_engine requires these
    parser.add_argument('--custom_unlearn', action='store_true')
    parser.add_argument('--oculoplastics', action='store_true')
    parser.add_argument('--selective_unlearn', action='store_true')
    parser.add_argument('--percent_to_forget', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )

    model = load_checkpoint(args.checkpoint_path, 'original_model', device=device)
    model.eval()

    print(f'Checkpoint: {args.checkpoint_path}')
    print(f'Target layer: {args.target_layer}')
    print()
    print(f'{"Seed":<8}{"Mean e_i":<12}{"Std e_i":<12}{"Median":<12}'
          f'{"Min":<10}{"Max":<10}{"% > 0.5":<10}')
    print('-' * 74)

    all_results = {}

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        trainset_full, testset, dataset = get_dataset(args.data_name, args.dataset_dir)
        val_size = int(len(trainset_full) * 0.1)
        train_size = len(trainset_full) - val_size
        trainset, valset = random_split(
            trainset_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        num_classes, _ = set_num_classes(args, dataset)
        total_forget_class = sum(1 for _, t in trainset if t == args.forget_class)
        num_forget = int(total_forget_class * args.percent_to_forget)

        train_forget_loader, train_remain_loader, \
        _, _, _, _, _, _, _, _, _, _, _, _, _ = dataloader_engine(
            args, trainset, valset, testset,
            combined_df=None,
            num_forget=num_forget,
            oculoplastics=args.oculoplastics,
            selective_unlearning=args.selective_unlearn,
        )

        # Compute per-class retain centroids at the target layer
        centroids = compute_retain_centroids(
            model, train_remain_loader,
            primary_layer=args.target_layer,
            num_classes=num_classes,
            device=device,
        )

        # Compute per-sample e_i for every forget sample
        all_es = []
        with torch.no_grad():
            for x, _ in train_forget_loader:
                x = x.to(device)
                raw_feats = get_intermediate_features(model, x, args.target_layer)
                normed_feats = prepare_features(raw_feats)
                e = compute_entanglement_scores(normed_feats, centroids)
                all_es.append(e.cpu().numpy())

        all_es = np.concatenate(all_es)
        pct_high = 100.0 * (all_es > 0.5).mean()

        all_results[seed] = {
            'mean': float(all_es.mean()),
            'std': float(all_es.std()),
            'median': float(np.median(all_es)),
            'min': float(all_es.min()),
            'max': float(all_es.max()),
            'pct_above_0.5': float(pct_high),
            'all_values': all_es.tolist(),
        }

        print(f'{seed:<8}{all_es.mean():<12.4f}{all_es.std():<12.4f}'
              f'{np.median(all_es):<12.4f}{all_es.min():<10.4f}'
              f'{all_es.max():<10.4f}{pct_high:<10.2f}')

    print('-' * 74)

    means = [all_results[s]['mean'] for s in args.seeds]
    print(f'\nAcross-seed variation in mean e_i: {np.std(means):.4f}')
    print(f'  Range: [{min(means):.4f}, {max(means):.4f}]')

    sorted_seeds = sorted(all_results.items(), key=lambda x: x[1]['mean'])
    print('\nSeeds ranked by mean initial e_i (low → high):')
    for seed, r in sorted_seeds:
        print(f'  Seed {seed}: mean={r["mean"]:.4f}, '
              f'% entangled (>0.5)={r["pct_above_0.5"]:.1f}%')

    output_file = f'initial_es_{args.data_name}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved to {output_file}')


if __name__ == '__main__':
    main()
    