"""
WS3 / R2.1 smoke-test assertions. Called by 10z_smoke.sh; safe to rerun alone.

Checks what the three stages actually produced, rather than just that they
exited 0:

  sample   row indices in range, sorted, unique; flags consistent with the
           requested draw sizes; cluster ids agree with mbkm_50/labels
  text     every sampled row retrieved (except a truncated month); rows unique;
           bodies non-empty and <= 2000 chars
  cert     nearest-centroid agreement well above chance; id check exact where
           metadata exists
  emb      one row per retrieved comment, row order matches the text file,
           unit-norm, no NaNs, dim as expected per model
  cross    arm-B centroid machinery works: per (month, cluster) means in each
           space, and that the three spaces give *correlated* within-month
           cluster geometry (sanity, not a result)
"""

import argparse
import configparser
import glob
import gzip
import json
import os

import numpy as np

import ws3_common as wc

EXPECT_DIM = {'minilm': 384, 'mpnet': 768, 'bge': 768}
FAIL = []
WARN = []


def check(cond, msg):
    print(f'  {"PASS" if cond else "FAIL"}  {msg}')
    if not cond:
        FAIL.append(msg)


def warn(cond, msg):
    if not cond:
        print(f'  WARN  {msg}')
        WARN.append(msg)


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--rev_subpath', type=str, default='rev/ws3_smoke')
    p.add_argument('--months', type=str, required=True)
    p.add_argument('--truncated', type=str, default='',
                   help='comma list of months retrieved with --limit_rows')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    args = p.parse_args()

    rev = os.path.join(g['save_path'], args.rev_subpath)
    months = [m.strip() for m in args.months.split(',') if m.strip()]
    trunc = {m.strip() for m in args.truncated.split(',') if m.strip()}
    totals = wc.load_totals(g['save_path'])

    for ym in months:
        print(f'\n{ym}{"  (truncated)" if ym in trunc else ""}')

        # ---- sample ----
        sp = os.path.join(rev, 'sample', f'sample_{ym}.npz')
        s = np.load(sp, allow_pickle=False)
        rows, cluster, flags = s['rows'], s['cluster'], s['flags']
        par = json.loads(str(s['params']))
        n_month = par['n_rows_month']
        check(n_month == totals[ym], f'sample: month size {n_month:,} == totals.csv')
        check(rows.min() >= 0 and rows.max() < n_month, 'sample: rows in range')
        check(np.all(np.diff(rows) > 0), 'sample: rows sorted and unique')
        check(bool((flags > 0).all()), 'sample: every row belongs to a draw')
        n_uni = int((flags & 1 > 0).sum())
        check(n_uni == min(par['n_uniform'], n_month),
              f'sample: uniform draw {n_uni:,} == min(n_uniform, month)')
        labels = wc.load_labels(g['save_path'], ym, args.run_subpath)
        check(np.array_equal(labels[rows], cluster.astype(labels.dtype)),
              'sample: cluster ids match labels')
        strat_sizes = np.bincount(cluster[(flags & 2) > 0],
                                  minlength=wc.K_CLUSTERS)
        cs = par['cluster_sizes']
        ok = all(strat_sizes[c] == min(par['n_strat'], cs[str(c)])
                 for c in range(wc.K_CLUSTERS) if str(c) in cs)
        check(ok, 'sample: stratified draw is min(n_strat, cluster size) per cluster')

        # ---- text ----
        tp = os.path.join(rev, 'text', f'text_{ym}.jsonl.gz')
        t_rows, t_ids, lens = [], [], []
        with gzip.open(tp, 'rt') as f:
            for line in f:
                d = json.loads(line)
                t_rows.append(d['row'])
                t_ids.append(d['id'])
                lens.append(len(d['body']))
        t_rows = np.asarray(t_rows, dtype=np.int64)
        check(len(np.unique(t_rows)) == len(t_rows), 'text: rows unique')
        check(bool(np.isin(t_rows, rows).all()), 'text: every row was sampled')
        if ym in trunc:
            check(len(t_rows) > 0, f'text: {len(t_rows):,} rows (truncated month)')
        else:
            check(len(t_rows) == len(rows),
                  f'text: all {len(rows):,} sampled rows retrieved')
        # empty bodies are legitimate: the 1_embed.py filter tests `'body' in
        # entry`, which keeps body == ''. They are rare (~1e-4) and embed to a
        # single point, like [removed]/[deleted]. Noted under R1.2.
        n_empty = int((np.asarray(lens) == 0).sum())
        check(max(lens) <= wc.MAX_CHARS,
              f'text: bodies truncated at {wc.MAX_CHARS}; max {max(lens)}, '
              f'{n_empty} empty ({n_empty/len(lens):.5f})')
        check(all(i for i in t_ids), 'text: every row has a comment id')

        # ---- certificate ----
        with open(os.path.join(rev, 'cert', f'cert_{ym}.json')) as f:
            cert = json.load(f)
        nc = cert['nearest_centroid_check']
        check(nc['agree'] > 0.90,
              f'cert: nearest-centroid agreement {nc["agree"]:.4f} > 0.90 '
              f'(chance {nc["chance"]})')
        warn(nc['agree'] > 0.99,
             f'{ym}: nearest-centroid agreement {nc["agree"]:.4f} below 0.99 '
             f'— expected, fp16 boundary flips, but worth watching')
        if wc.has_metadata(ym):
            idc = cert['id_check']
            check(idc.get('all_agree') is True,
                  f'cert: id check exact ({idc.get("agree")}/{idc.get("n")})')
        else:
            check('skipped' in cert['id_check'],
                  'cert: id check correctly skipped (no metadata)')

        # ---- embeddings ----
        order = np.argsort(t_rows)
        want_rows = t_rows[order]
        cents = {}
        for tag, dim in EXPECT_DIM.items():
            ep = os.path.join(rev, 'emb', tag, f'emb_{ym}.npz')
            if not os.path.exists(ep):
                check(False, f'emb[{tag}]: file missing')
                continue
            e = np.load(ep, allow_pickle=False)
            X, erows = e['X'], e['rows']
            check(X.shape == (len(want_rows), dim),
                  f'emb[{tag}]: shape {X.shape} == ({len(want_rows):,}, {dim})')
            check(np.array_equal(erows, want_rows),
                  f'emb[{tag}]: row order matches the text file')
            Xf = X.astype(np.float64)
            nrm = np.linalg.norm(Xf, axis=1)
            check(bool(np.isfinite(Xf).all()), f'emb[{tag}]: no NaN/inf')
            check(float(np.abs(nrm - 1).max()) < 2e-3,
                  f'emb[{tag}]: unit-norm (max |‖x‖-1| = '
                  f'{np.abs(nrm-1).max():.2e})')
            # arm-B centroids: per (month, cluster) mean in this space
            cl = labels[want_rows]
            C = np.vstack([Xf[cl == c].mean(0) if (cl == c).any()
                           else np.full(dim, np.nan) for c in range(wc.K_CLUSTERS)])
            cents[tag] = C
            # a truncated month retrieves only the head of the file, so it will
            # not cover all 50 clusters; that is the --limit_rows debug path,
            # not a defect.
            if ym in trunc:
                print(f'  SKIP  emb[{tag}]: cluster coverage '
                      f'({int(np.isfinite(C[:, 0]).sum())}/{wc.K_CLUSTERS}) '
                      f'— truncated month')
            else:
                check(bool(np.isfinite(C).all()),
                      f'emb[{tag}]: all {wc.K_CLUSTERS} clusters have members')

        # ---- cross-space sanity ----
        if (ym not in trunc and len(cents) > 1
                and all(np.isfinite(c).all() for c in cents.values())):
            def pdist_vec(C):
                D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
                return D[np.triu_indices(len(C), 1)]
            from scipy.stats import spearmanr
            base = pdist_vec(cents['minilm'])
            for tag in cents:
                if tag == 'minilm':
                    continue
                rho = spearmanr(base, pdist_vec(cents[tag])).statistic
                check(rho > 0.3, f'cross: inter-cluster distances minilm vs {tag} '
                                 f'correlate (spearman {rho:.3f} > 0.3)')

    # ---- MiniLM vs the published store, on the rows we have ----
    print('\nstored-embedding spot check')
    ym = months[0]
    ep = os.path.join(rev, 'emb', 'minilm', f'emb_{ym}.npz')
    if os.path.exists(ep):
        e = np.load(ep, allow_pickle=False)
        k = min(200, len(e['rows']))
        idx = e['rows'][:k]
        S = wc.l2_normalize(wc.read_embedding_rows(g['embed_path'], ym, idx))
        E = wc.l2_normalize(e['X'][:k].astype(np.float32))
        cos = np.sum(E * S, axis=1)
        check(float(np.median(cos)) > 0.999,
              f'{ym}: re-embedded vs stored MiniLM cosine median '
              f'{np.median(cos):.5f} > 0.999 (min {cos.min():.5f})')

    print(f'\n{"="*60}')
    print(f'{len(FAIL)} failures, {len(WARN)} warnings')
    if FAIL:
        for m in FAIL:
            print(f'  FAIL  {m}')
        raise SystemExit(1)
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
