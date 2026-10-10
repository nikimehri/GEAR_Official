"""Recovers methods missing from an hpo_search.py {name}_all_baselines_best.csv
by rebuilding their summary rows directly from their own already-written
{name}_{method}_best.csv files - no re-running needed.

This addresses exactly the failure mode main()'s old write-once-at-the-end
behavior could produce (fixed in this same change - see write_overall_csv()
in hpo_search.py): if one method's search crashed, EVERY method's results
from that run were discarded, even ones whose own {name}_{method}_best.csv
had already been written successfully. If you have those per-method files
but they never made it into the merged overall CSV, this recovers them
without spending any more compute.

Usage:
    python baselines/rebuild_overall_csv.py \\
        --name hpo_cifar10_resnet18 \\
        --methods neggrad,cfk,euk,scrub,delete \\
        --forget_acc_threshold 0.02
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hpo_search import write_overall_csv  # noqa: E402


def rebuild_one(name, method, forget_acc_threshold):
    best_csv = f"{name}_{method}_best.csv"
    if not os.path.exists(best_csv):
        print(f"[rebuild_overall_csv] {method}: {best_csv} not found - cannot recover, needs a real re-run.")
        return False

    with open(best_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[rebuild_overall_csv] {method}: {best_csv} has no rows - skipping.")
        return False

    selected = [r for r in rows if r['Selected'] == 'True']
    if len(selected) != 1:
        print(f"[rebuild_overall_csv] {method}: expected exactly 1 row with Selected=True in "
              f"{best_csv}, found {len(selected)} - skipping (file may be corrupted).")
        return False
    winner = selected[0]

    # Reconstruct forget_acc_threshold_used exactly as select_best_trial()
    # originally computed it: if the winner already cleared the requested
    # threshold, that's the threshold that was used; otherwise this was an
    # adaptive-fallback pick, where the effective threshold equals the
    # winner's own val_forget_acc by construction of that fallback branch.
    winner_val_forget_acc = float(winner['Val Forget Acc'])
    if winner_val_forget_acc <= forget_acc_threshold:
        threshold_used = forget_acc_threshold
    else:
        threshold_used = winner_val_forget_acc

    # tsne_path isn't stored in the per-trial best.csv (only the live
    # run_study() summary carried it) - reconstruct the expected path by
    # convention (run_study()'s winner-only t-SNE naming). Verify this file
    # actually exists before trusting it; if --tsne was off for the
    # original run, it won't.
    expected_tsne_path = f"{name}_{method}_{method}_WINNER_TSNE_ONLY_tsne.png"
    tsne_path = expected_tsne_path if os.path.exists(expected_tsne_path) else 'N/A'

    summary = {
        'method': method,
        'n_trials_completed': len(rows),
        'forget_acc_threshold_used': threshold_used,
        'val_forget_acc': winner['Val Forget Acc'],
        'val_retain_acc': winner['Val Retain Acc'],
        'test_forget_acc': winner['Test Forget Acc'],
        'test_retain_acc': winner['Test Retain Acc'],
        'hyperparameters': winner['Hyperparameters'],
        'optuna_params': winner['Optuna Params'],
        'tsne_path': tsne_path,
    }
    write_overall_csv(name, summary)
    print(f"[rebuild_overall_csv] {method}: recovered from {best_csv} "
          f"({len(rows)} trials, winner hyperparameters={json.loads(winner['Hyperparameters'])})")
    return True


def main():
    p = argparse.ArgumentParser(description="Rebuild missing rows in an hpo_search.py overall CSV "
                                             "from already-written per-method best.csv files")
    p.add_argument('--name', type=str, required=True,
                   help="Same --name the original hpo_search.py run used.")
    p.add_argument('--methods', type=str, required=True,
                   help="Comma-separated list of methods missing from {name}_all_baselines_best.csv.")
    p.add_argument('--forget_acc_threshold', type=float, default=0.02,
                   help="Must match the --forget_acc_threshold the original run used, so the "
                        "recovered threshold_used value is reconstructed correctly.")
    args = p.parse_args()

    recovered, failed = [], []
    for method in [m.strip() for m in args.methods.split(',')]:
        if rebuild_one(args.name, method, args.forget_acc_threshold):
            recovered.append(method)
        else:
            failed.append(method)

    print(f"\n[rebuild_overall_csv] Recovered {len(recovered)}/{len(recovered) + len(failed)}: {recovered}")
    if failed:
        print(f"[rebuild_overall_csv] Could not recover (need a real re-run): {failed}")


if __name__ == '__main__':
    main()
