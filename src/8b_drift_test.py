"""
WS1 / R1.1 steps 2-4: corrected drift test for every aligned topic group.

For each group (label != -1, at least --min_size time points after the
within-month rule) computes, under a Monte Carlo permutation null of temporal
exchangeability (shuffle the time order of the group's centroids):
  - lam : Lambda with the step covariance re-estimated for every ordering,
          in an r-dim projection (see --basis, --r, --r_mode)
  - d2  : squared endpoint displacement ||c_last - c_first||^2 in the full space
Both p-values are (1 + #{null >= obs}) / (B + 1).

Run from src/:
    python 8b_drift_test.py --within_month asis --basis global --r 3 --seed 0
"""

import argparse
import configparser
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import drift


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--within_month', choices=['asis', 'random', 'average'], default='asis')
    parser.add_argument('--basis', choices=['global', 'group'], default='global',
                        help='global PCA on all centroids, or each group\'s own PCA')
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--r_mode', choices=['fixed', 'adaptive', 'half'], default='fixed',
                        help='fixed: r for every group, test only T-2 >= r; '
                             'adaptive: r_k = min(r, T-2) (saturates the covariance '
                             'at small T); half: r_k = min(r, (T-2)//2), which keeps '
                             'at least 2 steps per estimated dimension')
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--min_size', type=int, default=5)
    parser.add_argument('--no_d2', action='store_true', help='skip the delta^2 test')
    parser.add_argument('--out_root', type=str, default='../output/ws1/8b')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    tag = args.tag or f'{args.within_month}_{args.basis}_{args.r_mode}_r{args.r}_s{args.seed}'
    out_path = os.path.join(args.out_root, tag)
    os.makedirs(out_path, exist_ok=True)
    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)

    t0 = time.time()
    C = drift.load_centroids(run_path, k=args.k)
    labels = drift.load_group_labels(align_path)
    assert C.shape[0] == labels.shape[0], (C.shape, labels.shape)

    rng = np.random.default_rng(args.seed)
    if args.basis == 'global':
        # PCA components are nested, so fitting once at r and slicing is equivalent
        Cp = PCA(n_components=args.r, svd_solver='full').fit_transform(C)

    rows = []
    for coi in np.unique(labels[labels >= 0]):
        rows_k, per = drift.group_rows(labels, coi, k=args.k,
                                       within_month=args.within_month, rng=rng)
        X = drift.trajectory(C, rows_k)
        T = X.shape[0]
        row = {'coi': int(coi), 'T': T, 'n_rows': int((labels == coi).sum()),
               'n_months': len(np.unique(per)),
               'first_period': int(per[0]), 'last_period': int(per[-1]),
               'disp': drift.displacement(X)}
        if T < args.min_size:
            continue

        if args.r_mode == 'fixed':
            r_k = args.r
        elif args.r_mode == 'adaptive':
            r_k = min(args.r, T - 2)
        else:
            r_k = max(1, min(args.r, (T - 2) // 2))
        row['r'] = r_k
        if T - 2 >= r_k:
            if args.basis == 'global':
                Xr = drift.trajectory(Cp[:, :r_k], rows_k)
            else:
                Xr = drift.group_basis(X, r_k)
            row['lam'], row['p_lam'] = drift.perm_test(Xr, drift.lam_batch,
                                                       n_perm=args.n_perm, rng=rng)
        if not args.no_d2:
            row['d2'], row['p_d2'] = drift.perm_test(X, drift.delta2_batch,
                                                     n_perm=args.n_perm, rng=rng)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_path, 'groups.csv'), index=False)

    summary = {'n_groups': len(df)}
    for col in ['p_lam', 'p_d2']:
        if col in df:
            s = df[col].dropna()
            summary |= {f'{col}_n_tested': len(s),
                        f'{col}_lt_05': int((s < 0.05).sum()),
                        f'{col}_lt_01': int((s < 0.01).sum())}
    params = vars(args) | {'run_path': run_path, 'align_path': align_path,
                           'tag': tag, 'summary': summary,
                           'elapsed_s': round(time.time() - t0, 1)}
    with open(os.path.join(out_path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(tag, json.dumps(summary), f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
