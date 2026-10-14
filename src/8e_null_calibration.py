"""
WS1 / R1.1 step 6: calibration of the drift test on the single-month null runs.

The runs under km/kmp*, km/kmp50 and mbkm_50/bootstrap cluster the SAME month
repeatedly (varying the init / batching / resample) and align the resulting
centroids exactly as the main pipeline does. The resample index plays the role
of the period, so there is no time: the centroids of an aligned group are
exchangeable by construction and the test should reject at about alpha.

Uses the locked R1.1 settings (group basis, r_k = min(r, (T-2)//2), as-is
order, B = 10,000). Run from src/:
    python 8e_null_calibration.py --seed 0
"""

import argparse
import configparser
import json
import os
import re
import time

import numpy as np
import pandas as pd

import drift

DEFAULT_RUNS = 'km/kmp,km/kmp50,mbkm_50/bootstrap'


def discover(run_path):
    """(month, n_periods) pairs that have both aligned labels and centroids."""
    align_dir = os.path.join(run_path, 'align')
    models_dir = os.path.join(run_path, 'models')
    if not (os.path.isdir(align_dir) and os.path.isdir(models_dir)):
        return []
    models = os.listdir(models_dir)
    out = []
    for f in sorted(os.listdir(align_dir)):
        m = re.fullmatch(r'align_model_(\d{4}-\d{2})_labels\.npz', f)
        if not m:
            continue
        ym = m.group(1)
        n = len([x for x in models if re.fullmatch(rf'model_cc_{ym}_\d+\.npz', x)])
        if n:
            out.append((ym, n))
    return out


def load_run(run_path, ym, n_periods):
    C = []
    for i in range(n_periods):
        with open(os.path.join(run_path, 'models', f'model_cc_{ym}_{i}.npz'), 'rb') as f:
            C.append(np.load(f)['cc'])
    k = C[0].shape[0]
    C = np.vstack(C)
    with open(os.path.join(run_path, 'align', f'align_model_{ym}_labels.npz'), 'rb') as f:
        labels = np.load(f)['labels']
    return C, labels, k


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=str, default=DEFAULT_RUNS)
    parser.add_argument('--r', type=int, default=5)
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--min_size', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--out_path', type=str, default='../output/ws1/8e_nullcal')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    rows = []
    for sub in args.runs.split(','):
        run_path = os.path.join(g['save_path'], sub)
        for ym, n_periods in discover(run_path):
            C, labels, k = load_run(run_path, ym, n_periods)
            if C.shape[0] != labels.shape[0]:
                print(f'skip {sub} {ym}: C {C.shape} vs labels {labels.shape}')
                continue
            for coi in drift.tested_groups(labels, min_size=args.min_size):
                rows_k, _ = drift.group_rows(labels, coi, k=k)
                X = drift.trajectory(C, rows_k)
                T = X.shape[0]
                r_k = max(1, min(args.r, (T - 2) // 2))
                lam, p_lam = drift.perm_test(drift.group_basis(X, r_k), drift.lam_batch,
                                             n_perm=args.n_perm, rng=rng)
                d2, p_d2 = drift.perm_test(X, drift.delta2_batch,
                                           n_perm=args.n_perm, rng=rng)
                rows.append({'run': sub, 'month': ym, 'n_periods': n_periods, 'k': k,
                             'coi': int(coi), 'T': T, 'r': r_k,
                             'lam': lam, 'p_lam': p_lam, 'd2': d2, 'p_d2': p_d2})
            print(f'{sub} {ym}: {n_periods} resamples, '
                  f'{len([r for r in rows if r["run"] == sub and r["month"] == ym])} groups '
                  f'({time.time()-t0:.0f}s)')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_path, 'groups.csv'), index=False)

    summary = (df.groupby(['run', 'month'])
                 .apply(lambda x: pd.Series({
                     'n_periods': x.n_periods.iloc[0], 'n_groups': len(x),
                     'median_T': x['T'].median(),
                     'rej_lam_05': (x.p_lam < 0.05).mean(),
                     'rej_lam_01': (x.p_lam < 0.01).mean(),
                     'rej_d2_05': (x.p_d2 < 0.05).mean(),
                     'rej_d2_01': (x.p_d2 < 0.01).mean()}), include_groups=False)
                 .reset_index())
    summary.to_csv(os.path.join(args.out_path, 'summary.csv'), index=False)

    overall = {'n_groups': len(df),
               'rej_lam_05': float((df.p_lam < 0.05).mean()),
               'rej_lam_01': float((df.p_lam < 0.01).mean()),
               'rej_d2_05': float((df.p_d2 < 0.05).mean()),
               'rej_d2_01': float((df.p_d2 < 0.01).mean())}
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump(vars(args) | {'basis': 'group', 'r_mode': 'half', 'within_month': 'asis',
                                'overall': overall, 'elapsed_s': round(time.time() - t0, 1)},
                  f, indent=2)
    print(summary.to_string(index=False))
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
