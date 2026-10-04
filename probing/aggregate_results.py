"""Collects per-checkpoint linear_probe.py result JSONs into one CSV table:
rows = (arch, dataset, forget_class) x method, columns = mean +/- 95% CI
over seeds for each metric.

Usage:
    python probing/aggregate_results.py --results_dir ./probing/results \
        --output_csv probing/probe_summary.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ['forget_class_probe_acc', 'retain_class_probe_acc', 'overall_probe_acc', 'ncc_forget_acc']


def _confidence_interval_95(values):
    """95% CI half-width via the t-distribution (appropriate for the small
    seed counts - typically 3-10 - this project uses), not a fixed 1.96
    normal-approximation constant. Returns (mean, half_width); half_width is
    NaN when there's only one value (no spread to estimate)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, float('nan')
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, float(t_crit * sem)


def load_all_results(results_dir: str) -> list:
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True)):
        with open(path, 'r') as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[aggregate_results] WARNING: skipping unparseable JSON: {path}")
    return results


def aggregate(results: list) -> list:
    """Groups by (arch, dataset, forget_class, method); returns one row dict
    per group with mean/CI for every metric present across its seeds."""
    groups = defaultdict(list)
    for r in results:
        cfg = r['config']
        key = (cfg['arch'], cfg['dataset'], cfg['forget_class'], r['method'])
        groups[key].append(r)

    rows = []
    for (arch, dataset, forget_class, method), group in sorted(groups.items()):
        row = {
            'arch': arch,
            'dataset': dataset,
            'forget_class': forget_class,
            'method': method,
            'n_seeds': len(group),
            'seeds': sorted({r['config']['seed'] for r in group}),
        }
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            if not values:
                row[f'{metric}_mean'] = None
                row[f'{metric}_ci95'] = None
                continue
            mean, ci = _confidence_interval_95(values)
            row[f'{metric}_mean'] = round(mean, 6)
            row[f'{metric}_ci95'] = round(ci, 6) if ci == ci else None  # NaN check
        rows.append(row)
    return rows


def write_csv(rows: list, output_csv: str) -> None:
    if not rows:
        print("[aggregate_results] No results found - nothing to write.")
        return
    fieldnames = ['arch', 'dataset', 'forget_class', 'method', 'n_seeds', 'seeds']
    for metric in METRICS:
        fieldnames += [f'{metric}_mean', f'{metric}_ci95']

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row['seeds'] = ','.join(str(s) for s in row['seeds'])
            writer.writerow(row)
    print(f"[aggregate_results] Wrote {len(rows)} rows to {output_csv}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate linear_probe.py result JSONs into a CSV")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory to search recursively for result JSONs")
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args(argv)

    results = load_all_results(args.results_dir)
    print(f"[aggregate_results] Loaded {len(results)} result JSON(s) from {args.results_dir}")
    rows = aggregate(results)
    write_csv(rows, args.output_csv)
    return rows


if __name__ == '__main__':
    main()
