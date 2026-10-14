"""
WS1 / R1.1: Fig 4a-style UMAP map of the significant topic groups.

Reads a per-group results CSV (from 8b_drift_test.py or 8c_minp_map.py) and
colours each group by whether its p-value is below --alpha.

Run from src/:
    python 8d_sig_map.py --results ../output/ws1/8b/asis_group_half_r5_s0/groups.csv \
        --p_col p_lam --fig_path ../refs/revision/figs/R1.1_driftmap_draft \
        --title 'Lambda, group basis, r_k = min(5, (T-2)//2)'
"""

import argparse
import configparser
import os

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
    parser.add_argument('--seed', type=int, default=0, help='unused; kept for consistency')
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--p_col', type=str, default='p_lam')
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--draft', action='store_true', default=True)
    parser.add_argument('--final', dest='draft', action='store_false')
    parser.add_argument('--fig_path', type=str, required=True)
    args = parser.parse_args()

    align_path = os.path.join(g['save_path'], args.run_subpath, args.align_subpath)
    labels = drift.load_group_labels(align_path)
    with open(os.path.join(align_path, 'cu2d.npz'), 'rb') as f:
        U = np.load(f)['Cu2d']
    df = pd.read_csv(args.results).set_index('coi')
    p = df[args.p_col].dropna()

    sig, nonsig = p.index[p < args.alpha], p.index[p >= args.alpha]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    m = np.isin(labels, nonsig)
    ax.scatter(U[m, 0], U[m, 1], s=10, c=GRAY, alpha=0.5, lw=0,
               label=f'not significant ({len(nonsig)})')
    m = np.isin(labels, sig)
    ax.scatter(U[m, 0], U[m, 1], s=14, c=BLUE, alpha=0.75, lw=0,
               label=f'p < {args.alpha} ({len(sig)})')

    for coi, (name, off) in ANNOTATE.items():
        if coi not in p.index:
            continue
        x = U[labels == coi].mean(axis=0)
        ax.annotate(f'{name}  p={p.loc[coi]:.3f}', x, xytext=off,
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
    title = f'{"DRAFT: " if args.draft else ""}{args.title or args.p_col}, {len(p)} tested groups'
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower left', markerscale=1.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(f'{args.fig_path}.{ext}', dpi=150)
    plt.close(fig)
    print(f'{len(sig)}/{len(p)} significant -> {args.fig_path}.{{png,pdf}}')


if __name__ == '__main__':
    main()
