"""
WS2 / R2.4 baseline: describe the published grouping and check the tools.

  1. Summary statistics of the published grouping (mbkm_50/align/max_100).
  2. HDBSCAN re-run on the stored UMAP embedding (cu.npz). If this reproduces
     the published labels, all seed variance in the alignment is UMAP's.
  3. The agreement battery (matching.py) on known cases: the published
     grouping against itself, and against a size-preserving shuffle.
  4. A seeded re-alignment at the published settings, run twice (determinism),
     and compared with WS3's arm B2 control, which used the same seed and
     settings.
  5. (--drift) WS1's locked statistic through ws2_common.drift_table on the
     published grouping, checked against WS1's locked B=1e5 run: T, delta and
     Lambda must match exactly; p only to Monte Carlo error.

Run from src/:
    python 9a_baseline.py --seed 0 --drift
"""

import argparse
import configparser
import json
import os
import time

import numpy as np
import pandas as pd

import drift
import matching as mt
import ws2_common as w2


def fmt(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def shuffle_groups(labels, rng):
    """Permute group labels among non-noise units (sizes and noise kept)."""
    out = labels.copy()
    m = labels >= 0
    out[m] = rng.permutation(labels[m])
    return out


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--align_subpath', type=str, default='align/max_100')
    p.add_argument('--ws1_results', type=str,
                   default='../output/ws1/8b/asis_group_half_r5_s0_B100k/groups.csv')
    p.add_argument('--ws3_control', type=str, default='rev/ws3/align/armB2_labels_control.npz',
                   help="WS3's seeded re-alignment of the published centroids (under save_path)")
    p.add_argument('--drift', action='store_true', help='also run step 5 (~4 min)')
    p.add_argument('--n_perm', type=int, default=100000)
    p.add_argument('--out_path', type=str, default='../output/ws2/9a')
    p.add_argument('--tab_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()

    t0 = time.time()
    os.makedirs(args.out_path, exist_ok=True)
    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)
    rng = np.random.default_rng(args.seed)
    res = {'params': {**vars(args), 'save_path': g['save_path'],
                      'umap': w2.UMAP_KW, 'hdbscan': w2.HDBSCAN_KW}}

    C = drift.load_centroids(run_path, k=args.k)
    pub = drift.load_group_labels(align_path)
    print(f'centroids {C.shape}, labels {pub.shape} ({time.time()-t0:.0f}s)')

    # 1. published summary
    s = w2.partition_summary(pub, k=args.k)
    s['amax_labels'] = int(pub.max())        # what 3_align.py printed as the count
    res['published'] = s
    print('\n1. published grouping:', json.dumps(fmt(s), indent=1))

    # 2. HDBSCAN on the stored embedding
    Cu = np.load(os.path.join(align_path, 'cu.npz'))['Cu']
    lab_cu = w2.cluster(Cu)
    r2 = {'identical': bool((lab_cu == pub).all()),
          'n_differ': int((lab_cu != pub).sum()),
          **mt.pair_scores(pub, lab_cu), **mt.noise_summary(pub, lab_cu)}
    res['hdbscan_on_stored_cu'] = r2
    print('\n2. HDBSCAN on stored cu.npz:', json.dumps(fmt(r2), indent=1))

    # 3. battery on known cases
    self_ = {**mt.pair_scores(pub, pub, rng=rng), **mt.noise_summary(pub, pub)}
    shuf = shuffle_groups(pub, rng)
    null_ = {**mt.pair_scores(pub, shuf, rng=rng), **mt.noise_summary(pub, shuf)}
    m2o = mt.match_many_to_one(pub, pub)
    res['battery_self'] = self_
    res['battery_shuffle'] = null_
    res['m2o_self'] = {'n': len(m2o), 'overlap_min': float(m2o.overlap.min()),
                       'frac_to_noise_max': float(m2o.frac_to_noise.max()),
                       'identity': bool((m2o.g_a == m2o.g_b).all())}
    print('\n3a. published vs itself:', json.dumps(fmt(self_), indent=1))
    print('3b. published vs shuffled groups:', json.dumps(fmt(null_), indent=1))
    print('3c. many-to-one, self:', res['m2o_self'])

    # 4. seeded re-alignment, twice, and against WS3's control
    t = time.time()
    lab_a = w2.align(C, args.seed)
    ta = time.time() - t
    lab_b = w2.align(C, args.seed)
    np.savez_compressed(os.path.join(args.out_path, f'labels_published_settings_s{args.seed}.npz'),
                        labels=lab_a)
    r4 = {'seconds_per_alignment': round(ta, 1),
          'deterministic': bool((lab_a == lab_b).all()),
          'summary': w2.partition_summary(lab_a, k=args.k),
          'vs_published': {**mt.pair_scores(pub, lab_a, rng=rng),
                           **mt.noise_summary(pub, lab_a)}}
    ctrl_f = os.path.join(g['save_path'], args.ws3_control)
    if os.path.exists(ctrl_f):
        ctrl = np.load(ctrl_f)['labels']
        r4['identical_to_ws3_control'] = bool((ctrl == lab_a).all())
        r4['vs_ws3_control'] = mt.pair_scores(ctrl, lab_a)
    res['realign'] = r4
    print(f'\n4. seeded re-alignment (seed {args.seed}):',
          json.dumps({k: fmt(v) if isinstance(v, dict) else v for k, v in r4.items()},
                     indent=1))

    # 5. drift wrapper vs WS1's locked run
    if args.drift:
        t = time.time()
        d = w2.drift_table(C, pub, k=args.k, n_perm=args.n_perm, seed=args.seed)
        d.to_csv(os.path.join(args.out_path, 'drift_published.csv'), index=False)
        w1 = pd.read_csv(args.ws1_results)
        j = d.merge(w1[['coi', 'T', 'disp', 'lam', 'p_lam']], on='coi',
                    suffixes=('', '_ws1'))
        r5 = {'seconds': round(time.time() - t, 1),
              'n_tested': len(d), 'n_ws1': len(w1), 'n_joined': len(j),
              'T_equal': bool((j['T'] == j['T_ws1']).all()),
              'delta_maxdiff': float((j.delta - j.disp).abs().max()),
              'lam_max_reldiff': float(((j.lam - j.lam_ws1).abs()
                                        / j.lam_ws1.abs().clip(lower=1e-12)).max()),
              'p_spearman': float(j.p_lam.corr(j.p_lam_ws1, method='spearman')),
              'p_max_absdiff': float((j.p_lam - j.p_lam_ws1).abs().max()),
              'n_p05': int((d.p_lam < 0.05).sum()), 'n_p05_ws1': int((w1.p_lam < 0.05).sum()),
              'n_p01': int((d.p_lam < 0.01).sum()), 'n_p01_ws1': int((w1.p_lam < 0.01).sum()),
              'n_q05': int((d.q_lam < 0.05).sum()),
              'n_q05_ws1': int((drift.bh_qvalues(w1.p_lam.values) < 0.05).sum())}
        res['drift_vs_ws1'] = r5
        print('\n5. drift wrapper vs WS1:', json.dumps(fmt(r5), indent=1))

    res['elapsed_s'] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_path, 'baseline.json'), 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    # SI-facing table: the published grouping and one seeded re-alignment
    rows = [{'grouping': 'published (max_100, unseeded)', **res['published']},
            {'grouping': f're-aligned, published settings, seed {args.seed}',
             **r4['summary']}]
    tab = pd.DataFrame(rows).drop(columns=['amax_labels'], errors='ignore')
    tab.to_csv(os.path.join(args.tab_path, 'R2.4_baseline_draft.csv'), index=False)
    print(f'\ndone ({res["elapsed_s"]}s) -> {args.out_path}/baseline.json, '
          f'{args.tab_path}/R2.4_baseline_draft.csv')


if __name__ == '__main__':
    main()
