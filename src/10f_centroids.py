"""
WS3 / R2.1 stage 4: per-month centroids and partitions in every embedding space.

Produces the artifacts the three arms consume; all metrics and tables are in
10g_compare.py.

Stage 0, once per month, from the MiniLM embeddings
    assignment of every sampled row to its nearest *published* centroid. This,
    not the `labels` file, defines cluster membership everywhere downstream.
    MBKM labels are nearest-centroid assignments, so the two agree (median
    1.0000 over 204 months, 2026-10-21) — except for 2018-03, where the
    published labels came from a different fit than the published centroids.
    Deriving membership from the centroids keeps every month consistent with
    the centroids the published alignment actually used, with no special case.

Per (model, month)
    arm B    one centroid per (month, cluster): the mean of that cluster's
             *stratified* sample in this space. Plus two disjoint half-samples
             (split-half reliability of delta) and the --ladder sizes
             (subsample convergence), both for the sufficiency argument.
    arm A    MiniBatchKMeans(k=50) on the *uniform* sample in this space, for
             --n_seeds seeds, saving centroids and the partition of the sampled
             rows. Seeds give the within-model instability floor that cross-model
             agreement has to be read against.

Run from src/ (after 10e_embed.py):
    python 10f_centroids.py --seed 0
    python 10f_centroids.py --seed 0 --months 2006-06,2008-06 --rev_subpath rev/ws3_smoke
"""

import argparse
import configparser
import json
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import ws3_common as wc

UNIFORM, STRAT, BEYOND = 1, 2, 4
MODELS = ['minilm', 'mpnet', 'bge']


def load_month(rev, tag, ym):
    """(rows, X float32, cluster, flags) with the sample aligned to the embeddings."""
    e = np.load(os.path.join(rev, 'emb', tag, f'emb_{ym}.npz'), allow_pickle=False)
    erows, X = e['rows'], e['X'].astype(np.float32)
    s = np.load(os.path.join(rev, 'sample', f'sample_{ym}.npz'), allow_pickle=False)
    srows, cluster, flags = s['rows'], s['cluster'], s['flags']
    if len(erows) == len(srows) and np.array_equal(erows, srows):
        return erows, X, cluster, flags
    # a truncated or partial retrieval: keep the intersection, in row order
    common, ie, isamp = np.intersect1d(erows, srows, assume_unique=True,
                                       return_indices=True)
    return common, X[ie], cluster[isamp], flags[isamp]


def cluster_means(X, assign, k, rows_mask=None):
    """(k x d) means and counts; NaN rows where a cluster has no members."""
    d = X.shape[1]
    C = np.full((k, d), np.nan, dtype=np.float32)
    n = np.zeros(k, dtype=np.int64)
    idx = np.arange(len(X)) if rows_mask is None else np.flatnonzero(rows_mask)
    a = assign[idx]
    for c in range(k):
        sel = idx[a == c]
        n[c] = len(sel)
        if len(sel):
            C[c] = X[sel].mean(axis=0)
    return C, n


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--models', type=str, default=','.join(MODELS))
    p.add_argument('--months', type=str, default='')
    p.add_argument('--n_seeds', type=int, default=2,
                   help='arm A k-means restarts per month per space; seed pairs '
                        'give the within-model instability floor')
    p.add_argument('--ladder', type=str, default='250,500,1000',
                   help='per-cluster subsample sizes for the convergence check. '
                        'A rung at or above the stratified draw size takes the '
                        'whole sample, so its distance to the full centroid is '
                        'exactly 0; the full draw is the reference, not a rung.')
    p.add_argument('--k', type=int, default=wc.K_CLUSTERS)
    p.add_argument('--n_init', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=10000)
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    rev = os.path.join(g['save_path'], args.rev_subpath)
    tags = [t.strip() for t in args.models.split(',') if t.strip()]
    ladder = [int(x) for x in args.ladder.split(',') if x]
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or sorted(f[len('emb_'):-len('.npz')] for f in
                        os.listdir(os.path.join(rev, 'emb', 'minilm'))
                        if f.startswith('emb_')))
    os.makedirs(os.path.join(rev, 'assign'), exist_ok=True)
    t0 = time.time()

    # ---------- stage 0: nearest published centroid, from MiniLM ----------
    print(f'stage 0: assignment to published centroids, {len(months)} months')
    for ym in months:
        out = os.path.join(rev, 'assign', f'assign_{ym}.npz')
        if os.path.exists(out) and not args.overwrite:
            continue
        rows, X, cluster, flags = load_month(rev, 'minilm', ym)
        cc = wc.load_centroids_month(g['save_path'], ym, args.run_subpath)
        cc = cc.astype(np.float32)
        # ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2; only the last two terms matter
        assign = (((X @ cc.T) * -2.0) + (cc ** 2).sum(1)[None, :]).argmin(1)
        assign = assign.astype(np.int16)
        agree = float((assign == cluster)[cluster >= 0].mean())
        with open(out + '.part', 'wb') as f:
            np.savez_compressed(f, rows=rows, assign=assign,
                                labels_file=cluster, flags=flags,
                                meta=json.dumps({
                                    'month': ym, 'agree_with_labels_file': agree,
                                    'n': int(len(rows)),
                                    'run_subpath': args.run_subpath}))
        os.replace(out + '.part', out)
        if agree < 0.99:
            print(f'  {ym}: agreement with the labels file is {agree:.4f} '
                  f'— expected for 2018-03 (see response.md), investigate otherwise')
    print(f'  done ({time.time()-t0:.1f}s)')

    # ---------- per (model, month) ----------
    for tag in tags:
        out_dir = os.path.join(rev, 'cc', tag)
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n[{tag}]')
        for ym in months:
            out = os.path.join(out_dir, f'cc_{ym}.npz')
            if os.path.exists(out) and not args.overwrite:
                print(f'  {ym}: exists, skipping')
                continue
            t = time.time()
            rows, X, _, flags = load_month(rev, tag, ym)
            a = np.load(os.path.join(rev, 'assign', f'assign_{ym}.npz'),
                        allow_pickle=False)
            if not np.array_equal(a['rows'], rows):
                raise RuntimeError(f'{tag} {ym}: rows differ from the assignment')
            assign = a['assign'].astype(np.int64)
            strat = (flags & STRAT) > 0
            uni = (flags & UNIFORM) > 0

            # ---- arm B: per (month, cluster) mean over the stratified sample ----
            C_B, n_B = cluster_means(X, assign, args.k, strat)

            # split-half: two disjoint halves of each cluster's stratified rows
            rng = np.random.default_rng([args.seed, int(ym[:4]), int(ym[5:])])
            h1 = np.zeros(len(X), dtype=bool)
            for c in range(args.k):
                sel = np.flatnonzero(strat & (assign == c))
                if len(sel) < 2:
                    continue
                take = rng.permutation(len(sel))[:len(sel) // 2]
                h1[sel[take]] = True
            C_h1, n_h1 = cluster_means(X, assign, args.k, h1)
            C_h2, n_h2 = cluster_means(X, assign, args.k, strat & ~h1)

            # ladder: first m stratified rows per cluster (shuffled, so random)
            C_lad = np.full((len(ladder), args.k, X.shape[1]), np.nan, np.float32)
            n_lad = np.zeros((len(ladder), args.k), dtype=np.int64)
            for c in range(args.k):
                sel = np.flatnonzero(strat & (assign == c))
                if not len(sel):
                    continue
                sel = sel[rng.permutation(len(sel))]
                for li, m in enumerate(ladder):
                    s = sel[:m]
                    n_lad[li, c] = len(s)
                    if len(s):
                        C_lad[li, c] = X[s].mean(axis=0)

            # ---- arm A: re-cluster the uniform sample in this space ----
            Xu = X[uni]
            A_cc = np.zeros((args.n_seeds, args.k, X.shape[1]), dtype=np.float32)
            A_lab = np.zeros((args.n_seeds, len(Xu)), dtype=np.int8)
            inertia = []
            for si in range(args.n_seeds):
                # n_init=1 matches the published run, which called
                # MiniBatchKMeans().partial_fit per chunk (a single init), and
                # is 1.5x faster; batch_size=10000 was the fastest setting at
                # production size (4.3 s for 200k x 768).
                km = MiniBatchKMeans(n_clusters=args.k,
                                     random_state=args.seed * 1000 + si,
                                     n_init=args.n_init,
                                     batch_size=args.batch_size)
                A_lab[si] = km.fit_predict(Xu).astype(np.int8)
                A_cc[si] = km.cluster_centers_
                inertia.append(float(km.inertia_))

            meta = {'month': ym, 'model': tag, 'seed': args.seed,
                    'k': args.k, 'n_seeds': args.n_seeds, 'ladder': ladder,
                    'n_rows': int(len(rows)), 'n_uniform': int(uni.sum()),
                    'n_strat': int(strat.sum()), 'dim': int(X.shape[1]),
                    'armA_inertia_last_batch': inertia,
                    'armA_n_init': args.n_init, 'armA_batch_size': args.batch_size,
                    'armB_empty_clusters': int((n_B == 0).sum())}
            with open(out + '.part', 'wb') as f:
                np.savez_compressed(
                    f, armB_cc=C_B, armB_n=n_B,
                    armB_half1=C_h1, armB_half1_n=n_h1,
                    armB_half2=C_h2, armB_half2_n=n_h2,
                    armB_ladder=C_lad, armB_ladder_n=n_lad,
                    armA_cc=A_cc, armA_labels=A_lab,
                    armA_rows=rows[uni], ladder=np.array(ladder),
                    meta=json.dumps(meta))
            os.replace(out + '.part', out)
            print(f'  {ym}: armB n/cluster med {int(np.median(n_B)):>5,} '
                  f'(empty {int((n_B == 0).sum())}) | armA on {uni.sum():>7,} rows '
                  f'x {args.n_seeds} seeds | {time.time()-t:5.1f}s', flush=True)

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
