"""
WS1 / R1.1 step 1: reproduce the published Table 1 and Fig 4 p-values.

Runs the published permutation test (viz.ipynb cell 75) on mbkm_50 +
align/max_100, seeded, and compares with the values printed in the v2
manuscript and in the saved notebook output.

Run from src/:
    python 8a_baseline.py --seed 0
"""

import argparse
import configparser
import json
import os
import time

import numpy as np
import pandas as pd

import drift

# values as printed in the v2 manuscript (Table 1, Fig 4 caption) and the saved
# viz.ipynb output (cell 77 bottom rows, cell 83). coi = group id in align/max_100.
PUBLISHED = [
    # coi, name, source, delta, lam, p
    (510, 'woman',     'Table 1', 0.032, 0.142, 0.901),
    (244, 'music',     'Table 1', 0.038, 0.052, 0.897),
    (146, 'team',      'Table 1', 0.044, 0.113, 0.942),
    (766, 'games',     'Table 1', 0.050, 0.061, 0.903),
    (425, 'delicious', 'Table 1', 0.054, 0.551, 0.339),
    (811, 'religion',  'Fig 4c',  np.nan, 8.29, 0.001),
]


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--n_components', type=int, default=45)
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--min_size', type=int, default=5)
    parser.add_argument('--out_path', type=str, default='../output/ws1/8a_baseline')
    args = parser.parse_args()

    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)
    os.makedirs(args.out_path, exist_ok=True)

    t0 = time.time()
    C = drift.load_centroids(run_path)
    labels = drift.load_group_labels(align_path)
    assert C.shape[0] == labels.shape[0], (C.shape, labels.shape)
    print(f'C {C.shape} {C.dtype}, labels {labels.shape} ({time.time()-t0:.1f}s)')

    rng = np.random.default_rng(args.seed)
    Cp = drift.published_pca(C, n_components=args.n_components, seed=args.seed)

    groups = drift.tested_groups(labels, min_size=args.min_size)
    n_groups_all = len(np.unique(labels[labels >= 0]))
    print(f'groups: {n_groups_all} non-noise, {len(groups)} with T >= {args.min_size}')

    rows = []
    for coi in groups:
        idx = np.where(labels == coi)[0]
        periods = idx // 50
        res = drift.perm_test_published(Cp[idx], n_perm=args.n_perm, rng=rng)
        ex = drift.perm_exact_published(Cp[idx])
        rows.append({
            'coi': int(coi),
            'T': len(idx),
            'n_months': len(np.unique(periods)),
            'first_period': int(periods[0]),
            'last_period': int(periods[-1]),
            'diam': drift.diameter(C[idx]),
            'disp': drift.displacement(C[idx]),
            'lam': res['lam'],
            'p': res['p'],
            'p_plus1': res['p_plus1'],
            'p_exact_gt': ex['p_gt'],
            'p_exact_ge': ex['p_ge'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_path, 'groups.csv'), index=False)

    # compare with published values
    comp = []
    for coi, name, src, d_pub, l_pub, p_pub in PUBLISHED:
        r = df.loc[df.coi == coi].iloc[0]
        comp.append({'coi': coi, 'name': name, 'source': src,
                     'disp_pub': d_pub, 'disp': round(r.disp, 3),
                     'lam_pub': l_pub, 'lam': round(r.lam, 3),
                     'p_pub': p_pub, 'p': r.p, 'p_exact_gt': round(r.p_exact_gt, 4),
                     'p_exact_ge': round(r.p_exact_ge, 4), 'T': r['T']})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(args.out_path, 'compare_published.csv'), index=False)

    summary = {
        'n_groups_nonnoise': int(n_groups_all),
        'n_tested': len(df),
        'n_p_lt_05': int((df.p < 0.05).sum()),
        'n_p_lt_01': int((df.p < 0.01).sum()),
        'n_exact_gt_lt_05': int((df.p_exact_gt < 0.05).sum()),
        'n_exact_ge_lt_05': int((df.p_exact_ge < 0.05).sum()),
        'max_abs_mc_minus_exact': float((df.p - df.p_exact_gt).abs().max()),
        'n_T_gt_months': int((df['T'] > df.n_months).sum()),
        'n_T_minus_2_lt_r': int((df['T'] - 2 < args.n_components).sum()),
    }
    params = vars(args) | {'run_path': run_path, 'align_path': align_path,
                           'summary': summary}
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    print(comp.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f'done ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
