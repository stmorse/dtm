"""
WS2 / R2.4: sensitivity of the topic grouping to UMAP/HDBSCAN parameters and
seeds, one parameter at a time around the published setting.

Every variant relabels the same 10,200 published centroids, so two groupings
are two labelings of identical units and need no matching for the partition
scores. Matching (by shared units, `matching.match_many_to_one`) is used only
to carry each published group's drift result over to a variant.

Per run (variant x seed) this writes
  - the partition summary (group count, noise, sizes, tested groups),
  - the battery vs the published grouping (`pair_scores` with null, and
    `noise_summary` beside it),
  - the same battery vs the published-setting run with the SAME UMAP seed
    (the parameter effect with the seed held fixed) and vs the same variant at
    the next seed (the variant's own seed spread),
  - the drift test on the variant's own groups (WS1's locked call, B=1e5):
    rates at p<0.05 / p<0.01, medians; q-counts are recorded but BH sets are
    never compared as sets,
  - the published -> variant group match joined to both drift results, for
    rank correlations over all matched groups.

UMAP layouts are computed once per (UMAP setting, seed); HDBSCAN-only variants
reuse the published-setting layout of the same seed. The no-UMAP baselines are
deterministic and run once.

Run from src/:
    OMP_NUM_THREADS=2 python 9b_align_grid.py --seed 0 --n_seeds 10 --workers 10
    python 9b_align_grid.py --summarize_only     # re-aggregate from saved runs
"""

import argparse
import configparser
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import drift
import matching as mt
import ws2_common as w2

# (variant name, parameter, value, umap overrides, hdbscan overrides)
BASE = ('published', 'none', '', {}, {})
VARIANTS = [BASE] + \
    [(f'n_neighbors={v}', 'n_neighbors', v, {'n_neighbors': v}, {}) for v in (10, 30, 50)] + \
    [(f'min_dist={v}', 'min_dist', v, {'min_dist': v}, {}) for v in (0.0, 0.25)] + \
    [(f'align_dim={v}', 'align_dim', v, {'n_components': v}, {}) for v in (5, 20)] + \
    [(f'min_cluster_size={v}', 'min_cluster_size', v, {}, {'min_cluster_size': v})
     for v in (5, 10)] + \
    [(f'min_samples={v}', 'min_samples', v, {}, {'min_samples': v}) for v in (1, 5)] + \
    [(f'max_cluster_size={v}', 'max_cluster_size', v, {}, {'max_cluster_size': v})
     for v in (20, 50, None)]
# no UMAP: HDBSCAN on the centroids directly (deterministic, one run each)
NO_UMAP = ['no_umap_cosine', 'no_umap_pca50']

_C = _PUB = _DPUB = _ARGS = None


def _init(C, pub, dpub, args):
    global _C, _PUB, _DPUB, _ARGS
    _C, _PUB, _DPUB, _ARGS = C, pub, dpub, args


def umap_key(v):
    return json.dumps(v[3], sort_keys=True)


def label_path(args, name, seed):
    return os.path.join(args.out_path, 'labels', f'{name}_s{seed}.npz')


def no_umap_labels(C, name):
    from sklearn.decomposition import PCA
    if name == 'no_umap_cosine':
        return w2.cluster(C, metric='cosine')
    if name == 'no_umap_pca50':
        return w2.cluster(PCA(n_components=50, svd_solver='full').fit_transform(C))
    raise ValueError(name)


def align_job(job):
    """All variants sharing one UMAP layout for one seed -> {name: labels}."""
    ukw, names, seed = job
    out = {}
    todo = [n for n in names if not os.path.exists(label_path(_ARGS, n, seed))]
    if todo:
        if names[0] in NO_UMAP:
            for n in todo:
                out[n] = no_umap_labels(_C, n)
        else:
            Cu = w2.embed(_C, seed, **json.loads(ukw))
            for n in todo:
                hkw = next(v[4] for v in VARIANTS if v[0] == n)
                out[n] = w2.cluster(Cu, **hkw)
        for n, lab in out.items():
            np.savez_compressed(label_path(_ARGS, n, seed), labels=lab)
    return seed, names


def load_labels(args, name, seed):
    return np.load(label_path(args, name, seed))['labels']


def scores(a, b, rng, prefix):
    d = {**mt.pair_scores(a, b, rng=rng), **mt.noise_summary(a, b)}
    return {f'{prefix}{k}': v for k, v in d.items()}


def drift_match(pub, lab, dpub, dvar):
    """Published -> variant match joined to both drift tables, over all
    published tested groups (never conditioned on significance)."""
    m = mt.match_many_to_one(pub, lab)
    m = m[m.g_a.isin(dpub.coi)]
    m = m.merge(dpub[['coi', 'T', 'delta', 'lam', 'p_lam']]
                .rename(columns={'coi': 'g_a', 'T': 'T_pub', 'delta': 'delta_pub',
                                 'lam': 'lam_pub', 'p_lam': 'p_pub'}), on='g_a')
    m = m.merge(dvar[['coi', 'T', 'delta', 'lam', 'p_lam']]
                .rename(columns={'coi': 'g_b', 'T': 'T_var', 'delta': 'delta_var',
                                 'lam': 'lam_var', 'p_lam': 'p_var'}),
                on='g_b', how='left')
    return m


def eval_job(job):
    """Score one (variant, seed) grouping."""
    name, seed = job
    args, pub, dpub = _ARGS, _PUB, _DPUB
    rng = np.random.default_rng([args.seed, seed])
    lab = load_labels(args, name, seed)
    v = next((v for v in VARIANTS if v[0] == name), None)
    r = {'variant': name, 'param': v[1] if v else 'no_umap',
         'value': str(v[2]) if v else name, 'seed': seed,
         **w2.partition_summary(lab, k=args.k)}
    r.update(scores(pub, lab, rng, 'pub_'))
    if v is not None and name != BASE[0]:
        r.update(scores(load_labels(args, BASE[0], seed), lab, rng, 'paired_'))
    if v is not None and args.n_seeds > 1:
        nxt = args.seeds[(args.seeds.index(seed) + 1) % len(args.seeds)]
        if os.path.exists(label_path(args, name, nxt)):
            r.update(scores(lab, load_labels(args, name, nxt), rng, 'seed_'))

    f = os.path.join(args.out_path, 'drift', f'{name}_s{seed}.csv')
    if os.path.exists(f):
        d = pd.read_csv(f)
    else:
        d = w2.drift_table(_C, lab, k=args.k, n_perm=args.n_perm, seed=args.seed)
        d.to_csv(f, index=False)
    r.update({'drift_n_tested': len(d),
              'drift_rate_p05': float((d.p_lam < 0.05).mean()),
              'drift_rate_p01': float((d.p_lam < 0.01).mean()),
              'drift_n_q05': int((d.q_lam < 0.05).sum()),
              'drift_median_delta': float(d.delta.median()),
              'drift_median_T': float(d['T'].median())})

    m = drift_match(pub, lab, dpub, d)
    m.insert(0, 'seed', seed)
    m.insert(0, 'variant', name)
    tested = m.dropna(subset=['p_var'])
    r.update({'match_n_pub_tested': len(m),
              'match_frac_to_tested': len(tested) / max(len(m), 1),
              'match_overlap_median': float(m.overlap.median()),
              'match_frac_to_noise_median': float(m.frac_to_noise.median())})
    if len(tested) > 2:
        for c in ('delta', 'lam', 'p'):
            r[f'match_spearman_{c}'] = spearmanr(tested[f'{c}_pub'],
                                                 tested[f'{c}_var']).statistic
        # the same over near-identical matches: separates "the statistic is
        # unstable" from "the group changed"
        st = tested[(tested.overlap >= args.strong) &
                    (tested.frac_to_noise <= 1 - args.strong)]
        r['match_n_strong'] = len(st)
        if len(st) > 2:
            for c in ('delta', 'lam', 'p'):
                r[f'strong_spearman_{c}'] = spearmanr(st[f'{c}_pub'],
                                                      st[f'{c}_var']).statistic
        # among published groups at p<0.01, how often is the match p<0.05,
        # against the variant's own base rate (enrichment, not set identity)
        s = tested[tested.p_pub < 0.01]
        e = mt.enrichment_test(int((s.p_var < 0.05).sum()), len(s),
                               r['drift_rate_p05'])
        r.update({f'carry_p01_{k}': e[k] for k in ('hit', 'tot', 'rate', 'enrichment', 'p')})
    return r, m


def summarize(args, runs, matches):
    """Median and range over seeds per variant; the SI-facing table."""
    runs.to_csv(os.path.join(args.tab_path, 'R2.4_grid_runs_draft.csv'), index=False)
    matches.to_csv(os.path.join(args.out_path, 'matches.csv'), index=False)
    cols = [('n_groups', 'groups'), ('noise_frac', 'noise'), ('n_tested', 'tested'),
            ('pub_ari', 'ARI vs pub'), ('pub_ami', 'AMI vs pub'),
            ('pub_purity_1to1', 'purity vs pub'),
            ('pub_jaccard_noise', 'noise Jacc vs pub'),
            ('paired_ari', 'ARI vs same-seed pub-setting'),
            ('seed_ari', 'ARI seed-vs-seed'),
            ('drift_rate_p05', 'drift rate p<.05'), ('drift_rate_p01', 'drift rate p<.01'),
            ('match_frac_to_tested', 'pub groups matched to tested'),
            ('match_spearman_delta', 'rho delta'), ('match_spearman_lam', 'rho Lambda'),
            ('match_spearman_p', 'rho p'),
            ('match_n_strong', 'n strong matches'),
            ('strong_spearman_delta', 'rho delta (strong)'),
            ('strong_spearman_lam', 'rho Lambda (strong)'),
            ('carry_p01_rate', 'pub p<.01 -> match p<.05'),
            ('carry_p01_enrichment', 'enrichment')]
    order = [v[0] for v in VARIANTS] + NO_UMAP
    rows = []
    for name in order:
        g = runs[runs.variant == name]
        if not len(g):
            continue
        row = {'variant': name, 'n_seeds': len(g)}
        for c, lab in cols:
            if c not in g or g[c].isna().all():
                row[lab] = ''
                continue
            x = g[c].dropna()
            row[lab] = (f'{x.median():.3f}' if len(x) == 1 else
                        f'{x.median():.3f} [{x.min():.3f}, {x.max():.3f}]')
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.tab_path, 'R2.4_grid_summary_draft.csv'), index=False)
    lines = ['| ' + ' | '.join(tab.columns) + ' |',
             '|' + '|'.join('---' for _ in tab.columns) + '|']
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in tab.values]
    note = (f'R2.4 alignment grid, one parameter at a time around the published '
            f'setting (n_neighbors 15, min_dist 0.1, align_dim 10, min_cluster_size 3, '
            f'min_samples = min_cluster_size, max_cluster_size 100). Median [min, max] '
            f'over {args.n_seeds} UMAP seeds. Agreement is on units non-noise in both '
            f'groupings (matching.pair_scores); noise Jaccard beside it. Drift: WS1 '
            f'locked statistic, B={args.n_perm}. rho = Spearman over all published '
            f'tested groups whose best-overlap match is tested in the variant.\n\n')
    with open(os.path.join(args.tab_path, 'R2.4_grid_summary_draft.md'), 'w') as f:
        f.write(note + '\n'.join(lines) + '\n')
    print(tab.to_string(index=False))


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0, help='base seed (drift RNG, nulls)')
    p.add_argument('--n_seeds', type=int, default=10, help='UMAP seeds seed..seed+n-1')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--n_perm', type=int, default=100000)
    p.add_argument('--variants', type=str, default='',
                   help='comma list of variant names (default: all)')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--align_subpath', type=str, default='align/max_100')
    p.add_argument('--pub_drift', type=str, default='../output/ws2/9a/drift_published.csv')
    p.add_argument('--strong', type=float, default=0.8,
                   help='overlap >= this and frac_to_noise <= 1-this = a near-identical match')
    p.add_argument('--summarize_only', action='store_true')
    p.add_argument('--out_path', type=str, default='../output/ws2/9b')
    p.add_argument('--tab_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()
    args.seeds = list(range(args.seed, args.seed + args.n_seeds))
    t0 = time.time()
    for d in ('labels', 'drift'):
        os.makedirs(os.path.join(args.out_path, d), exist_ok=True)
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump({**vars(args), 'save_path': g['save_path'], 'umap': w2.UMAP_KW,
                   'hdbscan': w2.HDBSCAN_KW,
                   'variants': [(v[0], v[3], v[4]) for v in VARIANTS] + NO_UMAP},
                  f, indent=1, default=str)

    if args.summarize_only:
        summarize(args, pd.read_csv(os.path.join(args.tab_path, 'R2.4_grid_runs_draft.csv')),
                  pd.read_csv(os.path.join(args.out_path, 'matches.csv')))
        return

    run_path = os.path.join(g['save_path'], args.run_subpath)
    C = drift.load_centroids(run_path, k=args.k)
    pub = drift.load_group_labels(os.path.join(run_path, args.align_subpath))
    if os.path.exists(args.pub_drift):
        dpub = pd.read_csv(args.pub_drift)
    else:
        dpub = w2.drift_table(C, pub, k=args.k, n_perm=args.n_perm, seed=args.seed)
    want = set(args.variants.split(',')) if args.variants else None
    variants = [v for v in VARIANTS if want is None or v[0] in want or v[0] == BASE[0]]
    no_umap = [n for n in NO_UMAP if want is None or n in want]

    # 1. alignments: one job per (UMAP setting, seed); no-UMAP once
    jobs = {}
    for v in variants:
        jobs.setdefault(umap_key(v), []).append(v[0])
    align_jobs = [(k_, names, s) for k_, names in jobs.items() for s in args.seeds]
    align_jobs += [('{}', [n], args.seed) for n in no_umap]
    ctx = get_context('spawn')          # numba (UMAP) and fork do not mix
    with ctx.Pool(args.workers, initializer=_init, initargs=(C, pub, dpub, args)) as pool:
        for i, (s, names) in enumerate(pool.imap_unordered(align_job, align_jobs)):
            print(f'  aligned {i+1}/{len(align_jobs)}: seed {s} {names} '
                  f'({time.time()-t0:.0f}s)', flush=True)

        # 2. scoring + drift
        eval_jobs = [(v[0], s) for v in variants for s in args.seeds] + \
                    [(n, args.seed) for n in no_umap]
        recs, ms = [], []
        for i, (r, m) in enumerate(pool.imap_unordered(eval_job, eval_jobs)):
            recs.append(r)
            ms.append(m)
            if (i + 1) % 10 == 0 or i + 1 == len(eval_jobs):
                print(f'  scored {i+1}/{len(eval_jobs)} ({time.time()-t0:.0f}s)', flush=True)

    runs = pd.DataFrame(recs).sort_values(['variant', 'seed'])
    summarize(args, runs, pd.concat(ms, ignore_index=True))
    print(f'done ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
