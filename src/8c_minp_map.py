"""
WS1 / R1.1 option (c), exploratory: min-p Lambda over several projection dims.

For each aligned topic group (label != -1, T >= --min_size) runs
drift.perm_test_minp over r in --rs (group's own PCA basis, members in row
order), writes per-group results, and draws a Fig 4a-style UMAP map of the
significant set.

Run from src/:
    python 8c_minp_map.py --seed 0
Outputs:
    ../output/ws1/8c_minp/{groups.csv,params.json}
    ../refs/revision/figs/R1.1_minp_map_draft.{png,pdf}
"""

import argparse
import configparser
import json
import os
import pickle
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift

# groups annotated on the map: coi -> (name, label offset in points)
ANNOTATE = {
    798: ('scientific', (-95, 22)),
    290: ('conspiracy', (20, 26)),
    949: ('government', (-110, -4)),
    1001: ('president', (40, 14)),
    999: ('republican', (18, -22)),
    774: ('laughing', (8, 6)),
    811: ('religion', (8, 6)),
    244: ('music', (8, 6)),
    146: ('team', (8, 6)),
}

BLUE, GRAY, INK, INK2, SURFACE = '#2a78d6', '#c9c8c3', '#0b0b0b', '#52514e', '#fcfcfb'


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--rs', type=str, default='2,3,5,10')
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--min_size', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--out_path', type=str, default='../output/ws1/8c_minp')
    parser.add_argument('--fig_path', type=str, default='../refs/revision/figs/R1.1_minp_map_draft')
    args = parser.parse_args()

    rs = [int(r) for r in args.rs.split(',')]
    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)

    t0 = time.time()
    C = drift.load_centroids(run_path, k=args.k)
    labels = drift.load_group_labels(align_path)
    with open(os.path.join(align_path, 'cu2d.npz'), 'rb') as f:
        U = np.load(f)['Cu2d']
    keywords = load_keywords(run_path)

    rng = np.random.default_rng(args.seed)
    rows = []
    for coi in drift.tested_groups(labels, min_size=args.min_size):
        rows_k, per = drift.group_rows(labels, coi, k=args.k)
        X = drift.trajectory(C, rows_k)
        T = X.shape[0]
        Xr = drift.group_basis(X, min(max(rs), T - 2))
        p, pr = drift.perm_test_minp(Xr, rs, n_perm=args.n_perm, rng=rng)
        idx = np.where(labels == coi)[0]
        top = Counter(w for i in idx for w in keywords[i][:10]).most_common(3)
        rows.append({'coi': int(coi), 'T': T,
                     'first_year': 2006 + int(per[0]) // 12,
                     'last_year': 2006 + int(per[-1]) // 12,
                     'disp': drift.displacement(X),
                     'kw': ' '.join(w for w, _ in top),
                     'p_minp': p,
                     **{f'p_r{r}': pr.get(r, np.nan) for r in rs}})
    df = pd.DataFrame(rows).set_index('coi')
    df.to_csv(os.path.join(args.out_path, 'groups.csv'))

    n_sig = int((df.p_minp < args.alpha).sum())
    params = vars(args) | {'rs': rs, 'basis': 'group', 'within_month': 'asis',
                           'run_path': run_path, 'align_path': align_path,
                           'n_tested': len(df), 'n_sig': n_sig,
                           'elapsed_s': round(time.time() - t0, 1)}
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    plot_map(U, labels, df, rs, args)
    print(f'{n_sig}/{len(df)} groups with p_minp < {args.alpha}; '
          f'figure -> {args.fig_path}.{{png,pdf}} ({time.time()-t0:.0f}s)')


def load_keywords(run_path, yrmo=None):
    """TF-IDF keyword lists, one per centroid row."""
    kw = []
    for year, month in (yrmo or drift.yrmo_range()):
        with open(os.path.join(run_path, 'tfidf', f'tfidf_{year}-{month}.pkl'), 'rb') as f:
            tf = pickle.load(f)['tfidf']
        kw += [tf[k]['keywords'] for k in range(len(tf))]
    return kw


def plot_map(U, labels, df, rs, args):
    sig = df.index[df.p_minp < args.alpha]
    nonsig = df.index[df.p_minp >= args.alpha]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    m = np.isin(labels, nonsig)
    ax.scatter(U[m, 0], U[m, 1], s=10, c=GRAY, alpha=0.5, lw=0,
               label=f'not significant ({len(nonsig)})')
    m = np.isin(labels, sig)
    ax.scatter(U[m, 0], U[m, 1], s=14, c=BLUE, alpha=0.75, lw=0,
               label=f'p < {args.alpha} ({len(sig)})')

    for coi, (name, off) in ANNOTATE.items():
        if coi not in df.index:
            continue
        x = U[labels == coi].mean(axis=0)
        ax.annotate(f'{name}  p={df.loc[coi, "p_minp"]:.3f}', x, xytext=off,
                    textcoords='offset points', fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=0, shrinkB=3))
        ax.plot(*x, marker='o', ms=5, mfc='none', mec=INK, mew=1)

    lo, hi = np.percentile(U[labels >= 0], [1, 99], axis=0)
    pad = (hi - lo) * 0.08
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    for s in ax.spines.values():
        s.set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel('UMAP 1', color=INK2)
    ax.set_ylabel('UMAP 2', color=INK2)
    ax.set_title(f'DRAFT: min-p Λ over r ∈ {{{",".join(map(str, rs))}}}, group basis, '
                 f'{len(df)} tested groups', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower left', markerscale=1.8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f'{args.fig_path}.{ext}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
