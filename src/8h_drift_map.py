"""
WS1 / R1.1 step 8b: revised Fig 4 from the per-group table.

Draws the UMAP map of all aligned topic-group centroids, coloured by whether
the group's BH-FDR q-value clears --alpha, and annotates a chosen set of
groups. Everything it needs about a group comes from the CSV written by
8g_group_table.py, so curation happens on the table, not in this script.

Which groups get names:
    --annotate auto:12          the 12 strongest surviving groups (default)
    --annotate list:811,1002    exactly these
    --annotate file:cois.txt    one coi per line (optionally "coi  label")
    --annotate published        the topics named in the published Table 1 / Fig 4
    --annotate none
Groups passed to --exclude are drawn in a third colour, dropped from the
significant count, and never annotated; the figure states how many, so a note
in the text can match it ("n groups excluded as platform artifacts").

Run from src/ (needs /mnt/sciclone mounted):
    python 8h_drift_map.py
    python 8h_drift_map.py --exclude 561,1002 --annotate auto:10 --final \
        --fig_path ../refs/revision/figs/R1.1_fig4_revised
    python 8h_drift_map.py --traj 811,774
"""

import argparse
import configparser
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift

BLUE, ORANGE, GRAY, INK, INK2, SURFACE = '#2a78d6', '#eb6834', '#c9c8c3', '#0b0b0b', '#52514e', '#fcfcfb'


def parse_cois(s):
    return [] if not s else [int(x) for x in s.replace(' ', '').split(',') if x]


def resolve_annotate(spec, df, sig, alpha):
    """-> {coi: label text} for the groups to name on the map."""
    spec = (spec or 'none').strip()
    if spec == 'none':
        return {}
    if spec == 'published':
        cois = [c for c in df.index if df.at[c, 'published']]
    elif spec.startswith('auto:'):
        cois = list(df.loc[sig].sort_values('q_lam', kind='stable').index[:int(spec[5:])])
    elif spec.startswith('list:'):
        cois = parse_cois(spec[5:])
    elif spec.startswith('file:'):
        cois, custom = [], {}
        for line in open(spec[5:]):
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            cois.append(int(parts[0]))
            if len(parts) > 1:
                custom[int(parts[0])] = parts[1]
        out = {c: custom.get(c) for c in cois if c in df.index}
        return {c: (v or default_label(df, c)) for c, v in out.items()}
    else:
        raise ValueError(f'--annotate: {spec}')
    return {c: default_label(df, c) for c in cois if c in df.index}


def default_label(df, coi):
    name = df.at[coi, 'name'] if 'name' in df else ''
    return name if isinstance(name, str) and name else str(df.at[coi, 'kw'])


def place_labels(pts, widths, xlim, ylim, pad=0.055, n_iter=400, seed=0):
    """Push label anchors apart, keeping each near its point.

    Works in axes-normalised coordinates so x and y are comparable. Labels
    start offset radially outward from the centre of mass, then repel each
    other and are pulled weakly back toward that starting offset.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([xlim[0], ylim[0]])
    span = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    P = (pts - lo) / span
    out = P.mean(axis=0)
    v = P - out
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(n > 1e-9, v / np.maximum(n, 1e-9), rng.normal(size=v.shape))
    target = P + v * pad * 1.6
    L = target + rng.normal(scale=1e-3, size=target.shape)
    w = np.asarray(widths, dtype=float)
    for _ in range(n_iter):
        F = (target - L) * 0.04                      # spring back to the anchor
        for i in range(len(L)):
            d = L[i] - L                             # repel other labels
            dist = np.hypot(d[:, 0] / np.maximum(w + w[i], 1e-9) * 2, d[:, 1] / pad)
            dist[i] = np.inf
            close = dist < 1.0
            if close.any():
                u = d[close] / np.maximum(np.linalg.norm(d[close], axis=1, keepdims=True), 1e-9)
                F[i] += (u * (1.0 - dist[close])[:, None] * 0.03).sum(axis=0)
            dp = L[i] - P[i]                         # never sit on top of the point
            r = np.hypot(dp[0], dp[1])
            if r < pad * 0.8:
                F[i] += dp / max(r, 1e-9) * (pad * 0.8 - r) * 0.5
        L = np.clip(L + F, 0.012, 0.988)
    return L * span + lo


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='label placement only')
    parser.add_argument('--table', type=str,
                        default='../refs/revision/tables/R1.1_groups_all_draft.csv',
                        help='per-group CSV from 8g_group_table.py')
    parser.add_argument('--q_col', type=str, default='q_lam',
                        help='significance column (q_lam, q_d2, or p_lam for uncorrected)')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--annotate', type=str, default='auto:12',
                        help='auto:N | list:a,b,c | file:path | published | none')
    parser.add_argument('--exclude', type=str, default='',
                        help='coi to draw as curated-out artifacts')
    parser.add_argument('--label_col', type=str, default='kw')
    parser.add_argument('--no_q', dest='show_q', action='store_false', default=True,
                        help='omit the q-value from each label')
    parser.add_argument('--offsets', type=str, default='',
                        help='JSON {"coi": [dx, dy]} in data units, overriding auto placement')
    parser.add_argument('--traj', type=str, default='',
                        help='coi list to add trajectory panels for (loads centroids)')
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--draft', action='store_true', default=True)
    parser.add_argument('--final', dest='draft', action='store_false')
    parser.add_argument('--fig_path', type=str,
                        default='../refs/revision/figs/R1.1_fig4_revised_draft')
    args = parser.parse_args()

    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)
    labels = drift.load_group_labels(align_path)
    with open(os.path.join(align_path, 'cu2d.npz'), 'rb') as f:
        U = np.load(f)['Cu2d']

    df = pd.read_csv(args.table).set_index('coi')
    if 'published' in df:
        df['published'] = df.published.fillna('')
    q = df[args.q_col].dropna()

    excl = [c for c in parse_cois(args.exclude) if c in q.index]
    sig = [c for c in q.index[q < args.alpha] if c not in excl]
    nonsig = [c for c in q.index[q >= args.alpha] if c not in excl]
    ann = {c: t for c, t in resolve_annotate(args.annotate, df, sig, args.alpha).items()
           if c not in excl}

    traj = [c for c in parse_cois(args.traj) if c in df.index]
    if traj:
        fig = plt.figure(figsize=(12.5, 7), facecolor=SURFACE)
        gs = fig.add_gridspec(len(traj), 2, width_ratios=[1.75, 1], wspace=0.2, hspace=0.42,
                              left=0.055, right=0.94, top=0.93, bottom=0.075)
        ax = fig.add_subplot(gs[:, 0])
        taxes = [fig.add_subplot(gs[i, 1]) for i in range(len(traj))]
    else:
        fig, ax = plt.subplots(figsize=(8.5, 7.2), facecolor=SURFACE)
        taxes = []

    ax.set_facecolor(SURFACE)
    m = np.isin(labels, nonsig)
    ax.scatter(U[m, 0], U[m, 1], s=10, c=GRAY, alpha=0.55, lw=0,
               label=f'not significant ({len(nonsig)})')
    m = np.isin(labels, sig)
    ax.scatter(U[m, 0], U[m, 1], s=16, c=BLUE, alpha=0.8, lw=0,
               label=f'{args.q_col} < {args.alpha} ({len(sig)})')
    if excl:
        m = np.isin(labels, excl)
        ax.scatter(U[m, 0], U[m, 1], s=18, c=ORANGE, alpha=0.9, lw=1.0,
                   marker='x', label=f'excluded as artifact ({len(excl)})')

    lo, hi = np.percentile(U[labels >= 0], [1, 99], axis=0)
    pad = (hi - lo) * 0.09
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if ann:
        cois = list(ann)
        texts = [f'{ann[c]}  q={q.loc[c]:.3f}' if args.show_q else ann[c] for c in cois]
        pts = np.array([U[labels == c].mean(axis=0) for c in cois])
        widths = [0.009 * len(t) for t in texts]
        L = place_labels(pts, widths, xlim, ylim, seed=args.seed)
        manual = {int(k): v for k, v in json.loads(open(args.offsets).read()).items()} \
            if args.offsets else {}
        for i, c in enumerate(cois):
            xy = pts[i]
            lxy = (xy[0] + manual[c][0], xy[1] + manual[c][1]) if c in manual else L[i]
            ha = 'left' if lxy[0] >= xy[0] else 'right'
            ax.annotate(texts[i], xy, xytext=lxy, textcoords='data', fontsize=8.5,
                        color=INK, ha=ha, va='center',
                        bbox=dict(boxstyle='round,pad=0.22', fc=SURFACE, ec=GRAY, lw=0.5, alpha=0.9),
                        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6,
                                        shrinkA=1, shrinkB=3))
            ax.plot(*xy, marker='o', ms=5, mfc='none', mec=INK, mew=1.1)

    for s in ax.spines.values():
        s.set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel('UMAP 1', color=INK2)
    ax.set_ylabel('UMAP 2', color=INK2)
    title = args.title or (f'Topic groups with BH-FDR {args.q_col} < {args.alpha} '
                           f'({len(q)} tested)')
    ax.set_title(('DRAFT: ' if args.draft else '') + title, fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower left', markerscale=1.8)

    if taxes:
        C = drift.load_centroids(run_path, k=args.k)
        for axt, coi in zip(taxes, traj):
            plot_trajectory(axt, C, labels, coi, df, q, args)

    if not taxes:
        fig.tight_layout()   # with the trajectory colorbars the gridspec is explicit
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(f'{args.fig_path}.{ext}', dpi=200)
    plt.close(fig)

    print(f'{len(sig)} significant, {len(nonsig)} not, {len(excl)} excluded, '
          f'{len(ann)} annotated -> {args.fig_path}.{{png,pdf}}')
    if ann:
        print('annotated: ' + ', '.join(f'{c} {ann[c]}' for c in ann))


def plot_trajectory(axt, C, labels, coi, df, q, args):
    """Group trajectory in its own top-2 principal axes, coloured by time."""
    rows_k, per = drift.group_rows(labels, coi, k=args.k)
    X = drift.trajectory(C, rows_k)
    Y = drift.group_basis(X, 2)
    axt.set_facecolor(SURFACE)
    axt.plot(Y[:, 0], Y[:, 1], '-', color=GRAY, lw=1, zorder=1)
    sc = axt.scatter(Y[:, 0], Y[:, 1], c=np.arange(len(Y)), cmap='viridis', s=26,
                     lw=0.3, edgecolor=SURFACE, zorder=2)
    axt.plot(Y[0, 0], Y[0, 1], marker='s', ms=7, mfc='none', mec=INK, mew=1.2, zorder=3)
    axt.plot(Y[-1, 0], Y[-1, 1], marker='*', ms=12, mfc='none', mec=INK, mew=1.2, zorder=3)
    lab = default_label(df, coi)
    axt.set_title(f'{lab} (coi {coi}): T={len(Y)}, {df.at[coi, "first_ym"]} to '
                  f'{df.at[coi, "last_ym"]}, {args.q_col}={q.loc[coi]:.3f}',
                  fontsize=8.5, color=INK, loc='left')
    for s in axt.spines.values():
        s.set_color(GRAY)
    axt.tick_params(colors=INK2, labelsize=7)
    axt.set_xlabel('group PC 1', color=INK2, fontsize=8)
    axt.set_ylabel('group PC 2', color=INK2, fontsize=8)
    cb = axt.figure.colorbar(sc, ax=axt, pad=0.02, fraction=0.045)
    cb.set_label('month index', color=INK2, fontsize=7)
    cb.ax.tick_params(colors=INK2, labelsize=6)


if __name__ == '__main__':
    main()
