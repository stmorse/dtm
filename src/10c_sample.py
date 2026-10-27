"""
WS3 / R2.1 stage 1 of 3: draw the sample index sets. Local, cheap, no raw text.

Per month, two draws from `mbkm_50/labels` (which covers every row of every
month, so no metadata is needed):

  uniform    n_uniform rows drawn uniformly over the month. Preserves cluster
             proportions, which arm A's re-clustering requires. Row indices are
             a bijection to kept comments, so uniform over indices is uniform
             over comments.
  stratified n_strat rows per (month, cluster). Gives every cluster the same
             centroid precision for arm B, including the small ones a uniform
             draw would leave noisy.

The union is what gets retrieved; membership flags say which draw each row
belongs to, so either analysis can recover its own sample exactly.

Run from src/:
    python 10c_sample.py --seed 0
    python 10c_sample.py --seed 0 --months 2006-06,2008-06    # smoke test
"""

import argparse
import configparser
import json
import os
import time

import numpy as np

import ws3_common as wc

UNIFORM = 1    # bit flags in the saved 'flags' array
STRAT = 2
BEYOND = 4     # row past the end of the published labels (truncated months)


def sample_month(labels, n_uniform, n_strat, rng, n_month=None):
    """
    (rows, cluster, flags, sizes) for one month, rows sorted ascending.

    n_month > len(labels) means the published labels cover only a prefix of the
    month (see the 2026-10-20 note in response.md: 4 months were clustered
    before their embeddings finished writing). The uniform draw then spans the
    whole month and rows past the labelled prefix get cluster -1 and the BEYOND
    flag; their cluster is recovered later by nearest published centroid, which
    reproduces the labels exactly. The stratified draw stays inside the prefix,
    since that is the only place cluster membership is known up front.
    """
    n = len(labels)
    n_draw = n if n_month is None else n_month

    # uniform draw. rng.choice(n, replace=False) would build a length-n
    # permutation (1.7 GB at n=215M), so oversample-and-dedupe instead: with
    # n_uniform << n collisions are rare.
    if n_uniform >= n_draw:
        uni = np.arange(n_draw, dtype=np.int64)
    else:
        need = int(n_uniform * 1.02) + 64
        uni = np.unique(rng.integers(0, n_draw, size=need, dtype=np.int64))
        while len(uni) < n_uniform:                       # vanishingly unlikely
            uni = np.unique(np.concatenate(
                [uni, rng.integers(0, n_draw, size=need, dtype=np.int64)]))
        uni = rng.permutation(uni)[:n_uniform]
        uni.sort()

    # stratified draw, per cluster
    strat = []
    sizes = {}
    for c in range(wc.K_CLUSTERS):
        rows_c = np.flatnonzero(labels == c)
        sizes[c] = int(len(rows_c))
        if len(rows_c) == 0:
            continue
        take = min(n_strat, len(rows_c))
        pick = rows_c if take == len(rows_c) else rng.choice(rows_c, size=take,
                                                            replace=False)
        strat.append(np.asarray(pick, dtype=np.int64))
    strat = np.unique(np.concatenate(strat)) if strat else np.empty(0, np.int64)

    rows = np.union1d(uni, strat)
    flags = np.zeros(len(rows), dtype=np.uint8)
    flags[np.isin(rows, uni, assume_unique=True)] |= UNIFORM
    flags[np.isin(rows, strat, assume_unique=True)] |= STRAT
    beyond = rows >= n
    flags[beyond] |= BEYOND
    cluster = np.full(len(rows), -1, dtype=np.int16)
    cluster[~beyond] = labels[rows[~beyond]].astype(np.int16)
    return rows, cluster, flags, sizes


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_uniform', type=int, default=200000)
    p.add_argument('--n_strat', type=int, default=2000)
    p.add_argument('--months', type=str, default='',
                   help='comma list; default all 204 months of the run')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--sample_full_month', action='store_true',
                   help='for months whose published labels cover only a prefix, '
                        'draw the uniform sample over the whole month; rows past '
                        'the prefix get cluster -1 and are labelled later by '
                        'nearest published centroid')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out_dir = os.path.join(g['save_path'], args.rev_subpath, 'sample')
    os.makedirs(out_dir, exist_ok=True)
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or wc.ym_range('2006-01', '2022-12'))
    totals = wc.load_totals(g['save_path'])
    t0 = time.time()
    summary = []

    for mi, ym in enumerate(months):
        out = os.path.join(out_dir, f'sample_{ym}.npz')
        if os.path.exists(out) and not args.overwrite:
            print(f'{ym}: exists, skipping')
            continue
        labels = wc.load_labels(g['save_path'], ym, args.run_subpath)
        n_month = totals.get(ym)
        truncated = n_month is not None and len(labels) != n_month
        if truncated:
            if len(labels) > n_month:
                raise RuntimeError(
                    f'{ym}: labels {len(labels):,} exceed totals {n_month:,}')
            print(f'{ym}: NOTE published labels cover {len(labels):,} of '
                  f'{n_month:,} rows ({len(labels)/n_month:.1%}) — clustered '
                  f'before the embeddings finished writing. '
                  f'{"Sampling the full month" if args.sample_full_month else "Sampling the labelled prefix only"}.')

        # seed per month, so a month is reproducible regardless of the --months list
        rng = np.random.default_rng([args.seed, mi, int(ym[:4]), int(ym[5:])])
        rows, cluster, flags, sizes = sample_month(
            labels, args.n_uniform, args.n_strat, rng,
            n_month=n_month if (truncated and args.sample_full_month) else None)
        np.savez_compressed(
            out, rows=rows, cluster=cluster, flags=flags,
            params=json.dumps({
                'month': ym, 'seed': args.seed, 'month_index': mi,
                'n_uniform': args.n_uniform, 'n_strat': args.n_strat,
                'n_rows_month': int(n_month if n_month else len(labels)),
                'n_rows_labels': int(len(labels)),
                'labels_truncated': bool(truncated),
                'sampled_full_month': bool(truncated and args.sample_full_month),
                'cluster_sizes': sizes,
                'uniform_bit': UNIFORM, 'strat_bit': STRAT, 'beyond_bit': BEYOND,
                'run_subpath': args.run_subpath,
            }))
        uni_cl = cluster[(flags & UNIFORM) > 0]
        per_cluster = np.bincount(uni_cl[uni_cl >= 0], minlength=wc.K_CLUSTERS)
        summary.append({
            'month': ym, 'n_rows_month': int(n_month if n_month else len(labels)),
            'n_rows_labels': int(len(labels)),
            'labels_truncated': bool(truncated),
            'n_beyond_labels': int(((flags & BEYOND) > 0).sum()),
            'n_sampled': int(len(rows)),
            'n_uniform': int(((flags & UNIFORM) > 0).sum()),
            'n_strat': int(((flags & STRAT) > 0).sum()),
            'uniform_per_cluster_min': int(per_cluster.min()),
            'uniform_per_cluster_median': float(np.median(per_cluster)),
            'frac_of_month': float(len(rows) / len(labels)),
        })
        n_draw = len(labels) if not (truncated and args.sample_full_month) \
            else n_month
        print(f'{ym}: month {n_draw:>12,} | sampled {len(rows):>8,} '
              f'({len(rows)/n_draw*100:5.2f}%) | uniform/cluster '
              f'min {per_cluster.min():>6,} med {np.median(per_cluster):>8,.0f} '
              f'({time.time()-t0:.1f}s)')

    if summary:
        # Rebuild the summary from every sample file on disk rather than
        # appending this run's rows: a resumed run skips completed months, so
        # appending leaves the summary covering only the last invocation.
        import csv
        sp = os.path.join(out_dir, 'summary.csv')
        allrows = []
        for f_ in sorted(glob.glob(os.path.join(out_dir, 'sample_*.npz'))):
            d = np.load(f_, allow_pickle=False)
            fl, cl = d['flags'], d['cluster']
            par = json.loads(str(d['params']))
            uni = (fl & UNIFORM) > 0
            uc = cl[uni]
            pc = np.bincount(uc[uc >= 0], minlength=wc.K_CLUSTERS)
            allrows.append({
                'month': par['month'],
                'n_rows_month': par['n_rows_month'],
                'n_rows_labels': par.get('n_rows_labels', par['n_rows_month']),
                'labels_truncated': par.get('labels_truncated', False),
                'n_sampled': int(len(fl)),
                'n_uniform': int(uni.sum()),
                'n_strat': int(((fl & STRAT) > 0).sum()),
                'n_both': int((uni & ((fl & STRAT) > 0)).sum()),
                'n_beyond_labels': int(((fl & BEYOND) > 0).sum()),
                'uniform_per_cluster_min': int(pc.min()),
                'uniform_per_cluster_median': float(np.median(pc)),
                'frac_of_month': float(len(fl) / par['n_rows_month']),
            })
        with open(sp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print(f'summary rebuilt from {len(allrows)} sample files')
        tot = sum(s['n_sampled'] for s in summary)
        print(f'\n{len(summary)} months, {tot:,} rows sampled '
              f'({tot/1e6:.1f}M). Summary -> {sp}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
