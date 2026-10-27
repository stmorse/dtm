"""
WS3 / R2.1 stage 5: the model-robustness comparisons. Produces the R2.1 tables.

Stages (--stages, default all; each is independent and writes its own table):

  agree   Partition agreement per month, from 10f's arm A partitions of the
          *same* uniform sample in each space. Three quantities that have to be
          read together:
            published vs MiniLM-sample  - the subset effect (sufficiency of the
                                          54.9M sample; the reviewer's question)
            seed i vs seed j, one space - the within-model instability floor
            MiniLM vs mpnet vs bge      - the model effect
          Cross-model agreement means little in the absolute; it means something
          against the floor. Comparison is by membership (ARI/AMI), never by
          centroid vectors, since the spaces have different dimension.

  armB1   Drift with the published partitions AND published groups inherited;
          only the coordinates change. A group is a list of (month, cluster)
          addresses, so its trajectory in another space is just those addresses'
          centroids in that space. Reports delta, Lambda, p, q per group per
          space, rank correlations across spaces, and the named groups.
          Includes MiniLM-sample as the sampling-noise control against the
          published centroids.

  armB2   Re-align the same 10,200 (month, cluster) units in each space with the
          published UMAP+HDBSCAN settings and compare the grouping to the
          published one. Units are identical, so plain ARI applies with no
          matching heuristic: the cleanest single test of whether the grouping
          survives a change of encoder.

  armA    The independent re-run: align each space's own k-means centroids,
          group them, and run the drift test. Groups do not correspond across
          spaces, so this compares distributions (group counts, noise fraction,
          delta, significant counts) plus a group correspondence built from
          member overlap.

  suff    Sufficiency detail: split-half reliability of delta (Spearman-Brown)
          and the subsample ladder, both from 10f's arm B artifacts.

Run from src/ (after 10f_centroids.py):
    python 10g_compare.py --seed 0
    python 10g_compare.py --seed 0 --stages agree,suff
"""

import argparse
import configparser
import json
import os
import time

# umap pulls a deprecated sklearn kwarg, and seeding it logs an n_jobs notice
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import homogeneity_completeness_v_measure as hcv

import drift
import ws3_common as wc

MODELS = ['minilm', 'mpnet', 'bge']
ALL_STAGES = ['agree', 'armB1', 'armB2', 'armA', 'suff']
# published alignment settings (3_align.py); random_state added per our rules
UMAP_KW = dict(n_neighbors=15, n_components=10, metric='euclidean',
               init='spectral', min_dist=0.1, spread=1.0)
HDBSCAN_KW = dict(min_cluster_size=3, min_samples=None,
                  cluster_selection_epsilon=0.0, max_cluster_size=100,
                  metric='euclidean')


def months_available(rev, tag='minilm'):
    d = os.path.join(rev, 'cc', tag)
    return sorted(f[len('cc_'):-len('.npz')] for f in os.listdir(d)
                  if f.startswith('cc_'))


def load_cc(rev, tag, ym):
    return np.load(os.path.join(rev, 'cc', tag, f'cc_{ym}.npz'), allow_pickle=False)


def stack_armB(rev, tag, months, k):
    """(n_months*k, d) arm B centroid matrix, row = period*k + cluster."""
    out = []
    for ym in months:
        out.append(load_cc(rev, tag, ym)['armB_cc'])
    return np.vstack(out).astype(np.float64)


def published_rows(months, k):
    """
    Row indices into the published 10,200-row centroid/label arrays for these
    months, and whether they are the complete set. A subset keeps the stages
    runnable for testing, but it truncates group trajectories, so it is not the
    analysis - the caller warns.
    """
    allm = [f'{y}-{m:02d}' for y in range(2006, 2023) for m in range(1, 13)]
    per = [allm.index(m) for m in months]
    rows = np.concatenate([np.arange(pp * k, (pp + 1) * k) for pp in per])
    return rows, len(months) == len(allm)


def load_published(args, months, k):
    """(centroids, group labels) for the selected months."""
    pub_labels = drift.load_group_labels(
        os.path.join(args.save_path, args.run_subpath, args.align_subpath))
    C_pub = drift.load_centroids(os.path.join(args.save_path, args.run_subpath),
                                 k=k)
    rows, complete = published_rows(months, k)
    if not complete:
        print(f'  WARNING: {len(months)} of 204 months. Group trajectories are '
              f'truncated to these months, so treat the output as a smoke test, '
              f'not the analysis.')
    return C_pub[rows], pub_labels[rows]


def align_units(C, seed, align_dim=10):
    """UMAP + HDBSCAN on a centroid matrix, published settings. -> labels."""
    import umap
    from sklearn.cluster import HDBSCAN
    kw = dict(UMAP_KW, n_components=align_dim, random_state=seed)
    Cu = umap.UMAP(**kw).fit_transform(C)
    model = HDBSCAN(**HDBSCAN_KW).fit(Cu)
    return model.labels_, Cu


def grouping_scores(a, b, rng=None, n_null=3):
    """
    Compare two groupings of the same units, separating the two things that an
    all-units ARI silently mixes together.

    HDBSCAN marks ~25-30% of units as noise (-1), and sklearn treats -1 as one
    cluster of ~2,800 units. Disagreement about *which* units are groupable then
    dominates the score: all-units ARI reads 0.11 where agreement on the units
    both groupings do place is ARI 0.52. Both numbers are real, so report both
    and never the first one alone.

    `*_core` is restricted to units that are non-noise in both. Homogeneity and
    completeness are not chance-adjusted and run ~0.69 under a size-preserving
    shuffle at this group count, so the null is returned with them; symmetric
    values mean boundary reshuffling rather than one grouping refining the other.
    """
    m = (a >= 0) & (b >= 0)
    h, c, _ = hcv(a[m], b[m])
    noise_a, noise_b = a == -1, b == -1
    out = {'ari_allunits': ari(a, b), 'ami_allunits': ami(a, b),
           'n_core': int(m.sum()),
           'ari_core': ari(a[m], b[m]), 'ami_core': ami(a[m], b[m]),
           'homogeneity': h, 'completeness': c,
           'noise_jaccard': float((noise_a & noise_b).sum() /
                                  max((noise_a | noise_b).sum(), 1)),
           'splits_per_group_a': float(np.mean(
               [len(np.unique(b[m][a[m] == g])) for g in np.unique(a[m])])),
           'splits_per_group_b': float(np.mean(
               [len(np.unique(a[m][b[m] == g])) for g in np.unique(b[m])]))}
    if rng is not None and n_null:
        nh, nc = [], []
        for _ in range(n_null):
            bs = rng.permutation(b[m])
            x, y, _ = hcv(a[m], bs)
            nh.append(x); nc.append(y)
        out['homogeneity_null'] = float(np.mean(nh))
        out['completeness_null'] = float(np.mean(nc))
    return out


def drift_table(C, labels, k, n_perm, seed, min_size=5):
    """delta/Lambda/p/q per tested group, using WS1's locked call."""
    rows = []
    for coi in drift.tested_groups(labels, min_size=min_size):
        X, per = drift.group_trajectory(C, labels, coi, k=k, within_month='asis')
        T = X.shape[0]
        r = max(1, min(5, (T - 2) // 2))
        lam, p = drift.perm_test(drift.group_basis(X, r), drift.lam_batch,
                                 n_perm=n_perm, rng=[seed, int(coi)])
        rows.append({'coi': int(coi), 'T': T, 'r': r,
                     'first_period': int(per[0]), 'last_period': int(per[-1]),
                     'delta': float(drift.displacement(X)),
                     'lam': float(lam), 'p_lam': float(p)})
    cols = ['coi', 'T', 'r', 'first_period', 'last_period', 'delta', 'lam',
            'p_lam', 'q_lam']
    if not rows:                      # no group reached min_size (small subsets)
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df['q_lam'] = drift.bh_qvalues(df.p_lam.values)
    return df[cols]


# ----------------------------------------------------------------- stages

def stage_agree(rev, months, tags, args, out):
    recs = []
    for ym in months:
        a = np.load(os.path.join(rev, 'assign', f'assign_{ym}.npz'),
                    allow_pickle=False)
        meta = json.loads(str(a['meta']))
        uni = (a['flags'] & 1) > 0
        pub_nearest = a['assign'][uni]
        pub_file = a['labels_file'][uni]
        lab = {}
        for tag in tags:
            d = load_cc(rev, tag, ym)
            lab[tag] = d['armA_labels']          # (n_seeds, n_uniform)
        n_seeds = lab[tags[0]].shape[0]
        rec = {'month': ym, 'n_uniform': int(uni.sum()), 'n_seeds': n_seeds,
               'assign_vs_labels_file': meta['agree_with_labels_file']}
        # subset effect: published partition vs the sample re-clustered in MiniLM
        if 'minilm' in lab:
            rec['ari_pub_minilm'] = ari(pub_nearest, lab['minilm'][0])
            rec['ami_pub_minilm'] = ami(pub_nearest, lab['minilm'][0])
            rec['ari_pubfile_minilm'] = ari(pub_file, lab['minilm'][0])
        # within-model floor
        for tag in tags:
            if n_seeds > 1:
                rec[f'ari_seed_{tag}'] = ari(lab[tag][0], lab[tag][1])
        # model effect
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                rec[f'ari_{t1}_{t2}'] = ari(lab[t1][0], lab[t2][0])
                rec[f'ami_{t1}_{t2}'] = ami(lab[t1][0], lab[t2][0])
        recs.append(rec)
        print(f'  {ym}: pub-minilm {rec.get("ari_pub_minilm", float("nan")):.3f} | '
              f'seed floor {rec.get("ari_seed_minilm", float("nan")):.3f} | '
              f'minilm-mpnet {rec.get("ari_minilm_mpnet", float("nan")):.3f}',
              flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_partition_agreement_draft.csv'), index=False)
    cols = [c for c in df.columns if c.startswith(('ari_', 'ami_'))]
    summ = df[cols].describe().T[['mean', '50%', 'min', 'max']]
    summ.to_csv(os.path.join(out, 'R2.1_partition_agreement_summary_draft.csv'))
    print('\npartition agreement over months (mean / median / min / max):')
    print(summ.round(4).to_string())
    return df


def stage_armB1(rev, months, tags, args, out):
    k = args.k
    C_pub, pub_labels = load_published(args, months, k)
    frames = {}
    frames['published'] = drift_table(C_pub, pub_labels, k, args.n_perm, args.seed)
    print(f'  published: {len(frames["published"])} groups tested')
    for tag in tags:
        C = stack_armB(rev, tag, months, k)
        bad = np.isnan(C).any(axis=1)
        if bad.any():
            # a cluster with no stratified members in some month; fall back to
            # the published centroid there so trajectories stay comparable
            C[bad] = C_pub[bad]
            print(f'  {tag}: {bad.sum()} of {len(C)} centroids had no sampled '
                  f'members; filled from the published centroid')
        frames[tag] = drift_table(C, pub_labels, k, args.n_perm, args.seed)
        n_sig = int((frames[tag].q_lam < 0.05).sum()) if len(frames[tag]) else 0
        print(f'  {tag}: {len(frames[tag])} groups tested, {n_sig} at q<0.05')
    if not len(frames['published']):
        print('  no group reached the minimum size; nothing to compare')
        return pd.DataFrame()
    # wide table
    base = frames['published'][['coi', 'T']].copy()
    for name, df in frames.items():
        base = base.merge(df[['coi', 'delta', 'lam', 'p_lam', 'q_lam']]
                          .rename(columns={c: f'{c}_{name}' for c in
                                           ['delta', 'lam', 'p_lam', 'q_lam']}),
                          on='coi', how='left')
    base.to_csv(os.path.join(out, 'R2.1_armB1_groups_draft.csv'), index=False)
    # cross-space agreement
    rows = []
    names = list(frames)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            for stat in ['delta', 'lam', 'p_lam']:
                x, y = base[f'{stat}_{n1}'], base[f'{stat}_{n2}']
                ok = x.notna() & y.notna()
                rows.append({'a': n1, 'b': n2, 'stat': stat,
                             'spearman': spearmanr(x[ok], y[ok]).statistic,
                             'n': int(ok.sum())})
            s1 = set(base.coi[base[f'q_lam_{n1}'] < 0.05])
            s2 = set(base.coi[base[f'q_lam_{n2}'] < 0.05])
            rows.append({'a': n1, 'b': n2, 'stat': 'sig_q05_jaccard',
                         'spearman': (len(s1 & s2) / len(s1 | s2)) if (s1 | s2) else np.nan,
                         'n': len(s1 | s2)})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(out, 'R2.1_armB1_agreement_draft.csv'), index=False)
    print('\narm B1 cross-space agreement:')
    print(cmp.round(4).to_string(index=False))
    return base


def stage_armB2(rev, months, tags, args, out):
    """
    Grouping and drift over the identical 10,200 (month, cluster) units.

    Two things are reported, as one ladder:

      grouping   ARI/AMI of each space's re-derived grouping against the
                 published one. Units are identical, so no matching is needed.
      drift      the WS1 statistic re-run on each *re-derived* grouping, in the
                 coordinates it was derived from. This is the cell arm B1 and
                 arm A between them leave empty: B1 changes coordinates with the
                 grouping held fixed, arm A changes partition and grouping and
                 coordinates at once. Here the partition is fixed and only the
                 grouping and coordinates move, so the question is specifically
                 whether re-grouping the same clusters preserves the drift
                 result.

    The `published_realigned` row is the control for both: same coordinates,
    same units, only the UMAP/HDBSCAN seed and code path differ. Whatever it
    loses is the cost of the alignment step alone, with no encoder involved.
    """
    k = args.k
    C_pub, pub_labels = load_published(args, months, k)
    # --out_path does not govern rev/align/, so a month subset would otherwise
    # overwrite the production labels with a truncated run's. Prefix instead.
    _, complete = published_rows(months, k)
    pre = '' if complete else f'smoke{len(months)}_'
    if not complete:
        print(f'  labels will be written as {pre}armB2_labels_*.npz, not over '
              f'the production ones')
    kw = None
    if args.keywords:
        kw = drift.load_keywords(os.path.join(args.save_path, args.run_subpath),
                                 yrmo=[(m[:4], m[5:]) for m in months])
    os.makedirs(os.path.join(rev, 'align'), exist_ok=True)

    # the grouping ladder: (name, labels, coordinates the drift is measured in)
    rungs = [('published', pub_labels, C_pub)]
    lab_ctrl, _ = align_units(C_pub, args.seed, args.align_dim)
    np.savez_compressed(os.path.join(rev, 'align', f'{pre}armB2_labels_control.npz'),
                        labels=lab_ctrl)
    rungs.append(('published_realigned', lab_ctrl, C_pub))
    for tag in tags:
        C = stack_armB(rev, tag, months, k)
        bad = np.isnan(C).any(axis=1)
        if bad.any():
            C[bad] = np.nanmean(C[~bad], axis=0) if C.shape[1] != C_pub.shape[1] \
                else C_pub[bad]
        lab, _ = align_units(C, args.seed, args.align_dim)
        np.savez_compressed(os.path.join(rev, 'align',
                                         f'{pre}armB2_labels_{tag}.npz'), labels=lab)
        rungs.append((tag, lab, C))

    recs = []
    for name, lab, C in rungs:
        t = time.time()
        d = drift_table(C, lab, k, args.n_perm, args.seed)
        ng = int(len(set(lab)) - (1 if -1 in lab else 0))
        sc = grouping_scores(pub_labels, lab, np.random.default_rng(args.seed))
        r = {'space': name, 'n_groups': ng,
             'noise_frac': float((lab == -1).mean()),
             'ari_vs_published': sc['ari_allunits'],
             'ami_vs_published': sc['ami_allunits'],
             'ari_core_vs_published': sc['ari_core'],
             'ami_core_vs_published': sc['ami_core'],
             'n_core': sc['n_core'],
             'homogeneity': sc['homogeneity'],
             'completeness': sc['completeness'],
             'homogeneity_null': sc['homogeneity_null'],
             'noise_jaccard': sc['noise_jaccard'],
             'splits_per_published_group': sc['splits_per_group_a'],
             'ari_core_vs_control': grouping_scores(lab_ctrl, lab)['ari_core'],
             'ari_vs_control': ari(lab_ctrl, lab),
             'n_tested': len(d),
             'median_T': float(d['T'].median()) if len(d) else np.nan,
             'median_delta': float(d['delta'].median()) if len(d) else np.nan,
             'median_lam': float(d['lam'].median()) if len(d) else np.nan,
             'n_p05': int((d.p_lam < 0.05).sum()) if len(d) else 0,
             'n_p01': int((d.p_lam < 0.01).sum()) if len(d) else 0,
             'n_q05': int((d.q_lam < 0.05).sum()) if len(d) else 0,
             'n_q01': int((d.q_lam < 0.01).sum()) if len(d) else 0}
        r['rate_p05'] = r['n_p05'] / r['n_tested'] if r['n_tested'] else np.nan
        r['rate_q05'] = r['n_q05'] / r['n_tested'] if r['n_tested'] else np.nan
        if kw is not None and len(d):
            # scale-free: is the drifting set keyword-coherent in every space?
            sig = d.coi[d.q_lam < 0.05].values
            r['sig_labels'] = ' | '.join(
                drift.group_label(kw, np.flatnonzero(lab == c), n=4)
                for c in sig[:12])
        recs.append(r)
        d.insert(0, 'space', name)
        d.to_csv(os.path.join(out, f'R2.1_armB2_drift_{name}_draft.csv'),
                 index=False)
        print(f'  {name:20s} {ng:5d} groups | ARI all-units '
              f'{r["ari_vs_published"]:.4f}, core {r["ari_core_vs_published"]:.4f} '
              f'(n={r["n_core"]:,}), noise jacc {r["noise_jaccard"]:.3f} | '
              f'{r["n_tested"]:4d} tested, {r["n_p05"]:4d} p<0.05 '
              f'({r["n_p05"]/max(r["n_tested"],1):.1%}) | {time.time()-t:5.1f}s',
              flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armB2_grouping_draft.csv'), index=False)
    print('\narm B2 grouping and drift (n_perm=%d):' % args.n_perm)
    cols = [c for c in df.columns if c != 'sig_labels']
    print(df[cols].round(4).to_string(index=False))
    return df


def stage_armA(rev, months, tags, args, out):
    k = args.k
    recs, labsets = [], {}
    for tag in tags:
        C = np.vstack([load_cc(rev, tag, ym)['armA_cc'][0] for ym in months])
        lab, _ = align_units(C.astype(np.float64), args.seed, args.align_dim)
        labsets[tag] = lab
        df = drift_table(C.astype(np.float64), lab, k, args.n_perm, args.seed)
        df.to_csv(os.path.join(out, f'R2.1_armA_groups_{tag}_draft.csv'),
                  index=False)
        recs.append({'space': tag, 'n_units': len(C),
                     'n_groups': int(len(set(lab)) - (1 if -1 in lab else 0)),
                     'noise_frac': float((lab == -1).mean()),
                     'n_tested': len(df),
                     'median_T': float(df['T'].median()) if len(df) else np.nan,
                     'median_delta': float(df.delta.median()) if len(df) else np.nan,
                     'n_sig_q05': int((df.q_lam < 0.05).sum()) if len(df) else 0,
                     'n_sig_q01': int((df.q_lam < 0.01).sum()) if len(df) else 0,
                     'frac_sig_q05': float((df.q_lam < 0.05).mean()) if len(df) else np.nan})
        print(f'  {tag}: {recs[-1]["n_groups"]} groups, {recs[-1]["n_tested"]} tested, '
              f'{recs[-1]["n_sig_q05"]} at q<0.05, median delta '
              f'{recs[-1]["median_delta"]:.3f}')
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armA_summary_draft.csv'), index=False)
    print('\narm A summary (independent re-run per space):')
    print(df.round(4).to_string(index=False))
    return df


def stage_suff(rev, months, tags, args, out):
    k = args.k
    recs = []
    for tag in tags:
        # split-half reliability of the per-(month,cluster) centroid, and of
        # delta over consecutive months as a trajectory-scale proxy
        d_half, d_lad = [], {}
        ladder = None
        for ym in months:
            z = load_cc(rev, tag, ym)
            h1, h2, nB = z['armB_half1'], z['armB_half2'], z['armB_n']
            ok = ~(np.isnan(h1).any(1) | np.isnan(h2).any(1))
            d_half.append(np.linalg.norm(h1[ok] - h2[ok], axis=1))
            ladder = z['ladder']
            full = z['armB_cc']
            for li, m in enumerate(ladder):
                sel = ok & ~np.isnan(z['armB_ladder'][li]).any(1)
                d_lad.setdefault(int(m), []).append(
                    np.linalg.norm(z['armB_ladder'][li][sel] - full[sel], axis=1))
        dh = np.concatenate(d_half)
        # half-sample centroids are each from n/2 points, so the distance
        # between them estimates sqrt(2) * SE(n/2) = 2 * SE(n)
        recs.append({'space': tag, 'quantity': 'half_vs_half_centroid_dist',
                     'n': int(len(dh)), 'median': float(np.median(dh)),
                     'p90': float(np.percentile(dh, 90)),
                     'implied_SE_full': float(np.median(dh) / 2)})
        for m, v in sorted(d_lad.items()):
            vv = np.concatenate(v)
            recs.append({'space': tag, 'quantity': f'ladder_{m}_vs_full',
                         'n': int(len(vv)), 'median': float(np.median(vv)),
                         'p90': float(np.percentile(vv, 90)),
                         'implied_SE_full': np.nan})
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_sufficiency_draft.csv'), index=False)
    print('\nsufficiency (centroid distances; compare with WS1 median step 0.139):')
    print(df.round(5).to_string(index=False))
    return df


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stages', type=str, default=','.join(ALL_STAGES))
    p.add_argument('--models', type=str, default=','.join(MODELS))
    p.add_argument('--months', type=str, default='')
    p.add_argument('--k', type=int, default=wc.K_CLUSTERS)
    p.add_argument('--n_perm', type=int, default=100000,
                   help='WS1 locked B=1e5: at 1e4 the smallest attainable p '
                        '(1e-4) exceeds BH tightest threshold (0.05/589=8.5e-5), '
                        'so the q<0.05 set is resolution-limited and unstable')
    p.add_argument('--keywords', action='store_true',
                   help='attach published TF-IDF labels to the B2 groups')
    p.add_argument('--align_dim', type=int, default=10)
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--align_subpath', type=str, default='align/max_100')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--out_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()
    args.save_path = g['save_path']

    rev = os.path.join(g['save_path'], args.rev_subpath)
    os.makedirs(os.path.join(rev, 'align'), exist_ok=True)
    os.makedirs(args.out_path, exist_ok=True)
    tags = [t.strip() for t in args.models.split(',') if t.strip()]
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or months_available(rev))
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    print(f'{len(months)} months, models {tags}, stages {stages}, seed {args.seed}')

    t0 = time.time()
    fns = {'agree': stage_agree, 'armB1': stage_armB1, 'armB2': stage_armB2,
           'armA': stage_armA, 'suff': stage_suff}
    for s in stages:
        if s not in fns:
            raise SystemExit(f'unknown stage {s}; choose from {ALL_STAGES}')
        print(f'\n=== {s} ===')
        fns[s](rev, months, tags, args, args.out_path)
    with open(os.path.join(args.out_path, 'R2.1_params.json'), 'w') as f:
        json.dump({**vars(args), 'n_months': len(months), 'models': tags,
                   'umap': UMAP_KW, 'hdbscan': HDBSCAN_KW}, f, indent=2,
                  default=str)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min -> {args.out_path}')


if __name__ == '__main__':
    main()
