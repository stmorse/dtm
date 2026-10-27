"""
WS3 / R2.8: the single-window null, re-run properly on several windows.

The S2 experiment takes one time window, produces R versions of it, treats those
R versions as if they were R time periods, runs the whole pipeline on them, and
checks that the drift test finds nothing. Any drift it does find is manufactured
by the pipeline, so the rejection rate estimates the false-positive rate.

Three things are wrong or missing in the published version, and this script
fixes all three while *reproducing* the original rather than replacing it:

 1. S2 says the window is "resampled with replacement". The code
    (5a_dist_singlemonth.py:121) calls np.random.permutation -- without
    replacement -- and a permutation has no effect anyway, because a single
    MiniBatchKMeans.partial_fit updates centers as a weighted mean over assigned
    points, which is order-invariant. The only thing that actually differs
    between resamples is the k-means++ initialization, which the S3 Fig caption
    describes correctly even though the S2 body text attributes it to batching.
    --nulls below separates the two sources instead of conflating them.

 2. The existing runs do not use the published alignment. Audited 2026-10-27:
    the paper's run used min_cluster_size=3, max_cluster_size=100, while every
    null run used 5 and 300 (bootstrap/2007-02 used 205; bootstrap/2007-05 used
    min_cluster_size=50). So S2 does not validate the pipeline as published, and
    G cannot be compared across those runs. This script uses wc.HDBSCAN_KW.

 3. The number of recovered groups G depends strongly on R, so a null at R=20 is
    not comparable to a 204-period analysis. At R=20 the existing runs give
    G = 48-73 with median T ~ R, exactly as S2 predicts (G ~ k); at R=204 they
    give G = 339-408 with median T = 13. Default --n_resamples is therefore 204,
    matching the real pipeline's period count.

Null modes (--nulls):
    init   same data every resample, k-means++ re-seeded. Reproduces S2.
    data   bootstrap resample of the rows WITH replacement, one shared fixed
           initial centroid matrix across all resamples. This is what S2 claims.
    both   bootstrap resample and a fresh init. The realistic case.

Data: the R2.1 uniform 200k/month sample, already embedded
(rev/ws3/emb/minilm/emb_YYYY-MM.npz), so no full-month read is needed. The
permutation test is exact under exchangeability at any n, and a smaller n means
noisier centroids, i.e. a harder test -- use --ladder to show the rejection rate
is flat in n.

Run from src/ (after 10e_embed.py):
    python 10k_nullrun.py --seed 0
    python 10k_nullrun.py --seed 0 --months 2015-01 --n_resamples 20 --n_perm 1000
"""

# umap 0.5.7 passes sklearn's old force_all_finite kwarg, and seeding UMAP logs
# an n_jobs notice. Both are cosmetic; seeding is deliberate (reproducible
# alignment) and costs UMAP's parallelism, which is not the bottleneck here.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')

import argparse
import configparser
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

import ws3_common as wc

UNIFORM = 1
NULLS = ['init', 'data', 'both']
# early / middle / late; five of the six overlap an existing run so the old and
# new nulls can be compared on the same window
WINDOWS = ['2007-02', '2010-02', '2015-01', '2017-03', '2020-06', '2021-06']


def load_uniform(rev, ym, tag='minilm'):
    """The month's uniform-draw rows, float32 unit-norm."""
    e = np.load(os.path.join(rev, 'emb', tag, f'emb_{ym}.npz'), allow_pickle=False)
    erows, X = e['rows'], e['X'].astype(np.float32)
    s = np.load(os.path.join(rev, 'sample', f'sample_{ym}.npz'), allow_pickle=False)
    srows, flags = s['rows'], s['flags']
    keep = (flags & UNIFORM) > 0
    want = srows[keep]
    if len(erows) != len(srows) or not np.array_equal(erows, srows):
        _, ie, _ = np.intersect1d(erows, want, assume_unique=True,
                                  return_indices=True)
        return X[ie]
    return X[keep]


def resample_centroids(X, mode, n_resamples, k, rng, seed, batch_size,
                       n_sub=None):
    """
    (R*k, d) centroids, resample r occupying rows r*k:(r+1)*k.

    `init` holds the data fixed and varies only the k-means++ draw, which is
    what the published experiment does. `data` holds the initialization fixed
    (one shared C0, computed once) and bootstraps the rows with replacement,
    which is what S2 describes. `both` varies both.
    """
    n = len(X) if n_sub is None else min(n_sub, len(X))
    C0 = None
    if mode == 'data':
        # one shared initialization for every resample, so the only thing that
        # varies is the data. Drawn from the full sample, not a resample of it.
        C0, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
    out = np.empty((n_resamples * k, X.shape[1]), dtype=np.float32)
    for r in range(n_resamples):
        if mode == 'init':
            Xr = X if n_sub is None else X[rng.choice(len(X), n, replace=False)]
        else:
            Xr = X[rng.integers(0, len(X), size=n)]      # WITH replacement
        km = MiniBatchKMeans(n_clusters=k, n_init=1, batch_size=batch_size,
                             init=(C0 if C0 is not None else 'k-means++'),
                             random_state=(seed if mode == 'data'
                                           else seed * 100000 + r))
        km.fit(Xr)
        out[r * k:(r + 1) * k] = km.cluster_centers_
    return out


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--months', type=str, default=','.join(WINDOWS))
    p.add_argument('--nulls', type=str, default=','.join(NULLS))
    p.add_argument('--n_resamples', type=int, default=204,
                   help='R. Default matches the real run\'s 204 periods; G '
                        'depends strongly on R, so a smaller R is not '
                        'comparable to the published analysis.')
    p.add_argument('--k', type=int, default=wc.K_CLUSTERS)
    p.add_argument('--n_perm', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=10000)
    p.add_argument('--align_dim', type=int, default=10)
    p.add_argument('--ladder', type=str, default='',
                   help='comma list of subsample sizes n, e.g. 50000,100000; '
                        'empty uses the whole uniform draw. Shows the rejection '
                        'rate is flat in n.')
    p.add_argument('--tag', type=str, default='',
                   help='suffix for the output tables, so a second run (e.g. '
                        'the --ladder control) does not overwrite the first')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--out_path', type=str, default='../refs/revision/tables')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    rev = os.path.join(g['save_path'], args.rev_subpath)
    months = [m.strip() for m in args.months.split(',') if m.strip()]
    nulls = [x.strip() for x in args.nulls.split(',') if x.strip()]
    ladder = [int(x) for x in args.ladder.split(',') if x.strip()] or [None]
    os.makedirs(args.out_path, exist_ok=True)
    cc_dir = os.path.join(rev, 'null')
    os.makedirs(cc_dir, exist_ok=True)
    t0 = time.time()

    recs, groups = [], []
    for ym in months:
        X = load_uniform(rev, ym)
        print(f'\n[{ym}] {len(X):,} uniform rows, dim {X.shape[1]}', flush=True)
        for mode in nulls:
            for n_sub in ladder:
                tag = f'{ym}_{mode}' + (f'_n{n_sub}' if n_sub else '')
                t = time.time()
                rng = np.random.default_rng([args.seed, hash(tag) % (2 ** 31)])
                C = resample_centroids(X, mode, args.n_resamples, args.k, rng,
                                       args.seed, args.batch_size, n_sub)
                lab, _ = wc.align_units(C, args.seed, args.align_dim)
                np.savez_compressed(os.path.join(cc_dir, f'null_{tag}.npz'),
                                    cc=C, labels=lab)
                d = wc.drift_table(C, lab, args.k, args.n_perm, args.seed)
                G = int(len(set(lab)) - (1 if -1 in lab else 0))
                r = {'month': ym, 'null': mode,
                     'n_rows': int(n_sub or len(X)),
                     'R': args.n_resamples, 'k': args.k,
                     'G': G, 'noise_frac': float((lab == -1).mean()),
                     'n_tested': len(d),
                     'median_T': float(d['T'].median()) if len(d) else np.nan,
                     'median_delta': float(d.delta.median()) if len(d) else np.nan,
                     'rej_p05': float((d.p_lam < 0.05).mean()) if len(d) else np.nan,
                     'rej_p01': float((d.p_lam < 0.01).mean()) if len(d) else np.nan,
                     'n_q05': int((d.q_lam < 0.05).sum()) if len(d) else 0,
                     'n_q01': int((d.q_lam < 0.01).sum()) if len(d) else 0,
                     'secs': round(time.time() - t, 1)}
                recs.append(r)
                if len(d):
                    d.insert(0, 'null', mode); d.insert(0, 'month', ym)
                    d.insert(2, 'n_rows', r['n_rows'])
                    groups.append(d)
                print(f'  {mode:5s} n={r["n_rows"]:>7,} R={args.n_resamples} | '
                      f'G={G:4d} noise={r["noise_frac"]:.3f} | '
                      f'{r["n_tested"]:4d} tested, median T {r["median_T"]:.0f}, '
                      f'reject@.05 {r["rej_p05"]:.4f} (@.01 {r["rej_p01"]:.4f}) | '
                      f'{r["secs"]:.0f}s', flush=True)

    s = pd.DataFrame(recs)
    sfx = f'_{args.tag}' if args.tag else ''
    s.to_csv(os.path.join(args.out_path,
                          f'R2.8_nullrun_summary{sfx}_draft.csv'), index=False)
    if groups:
        pd.concat(groups).to_csv(
            os.path.join(args.out_path,
                         f'R2.8_nullrun_groups{sfx}_draft.csv'), index=False)
    json.dump({'seed': args.seed, 'n_resamples': args.n_resamples,
               'k': args.k, 'n_perm': args.n_perm, 'align_dim': args.align_dim,
               'batch_size': args.batch_size, 'months': months, 'nulls': nulls,
               'ladder': [x for x in ladder if x],
               'umap': wc.UMAP_KW, 'hdbscan': wc.HDBSCAN_KW},
              open(os.path.join(args.out_path,
                                f'R2.8_params{sfx}.json'), 'w'),
              indent=2, default=str)
    print('\n' + s.to_string(index=False))
    if len(s):
        pooled = s[['n_tested']].sum().iloc[0]
        wavg = (s.rej_p05 * s.n_tested).sum() / max(pooled, 1)
        print(f'\npooled over {len(s)} runs: {pooled:,} tested groups, '
              f'rejection at .05 = {wavg:.4f} (nominal 0.05)')
    print(f'Done in {(time.time()-t0)/60:.1f} min -> {args.out_path}')


if __name__ == '__main__':
    main()
