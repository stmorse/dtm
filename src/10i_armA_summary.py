"""
WS3 / R2.1 arm A: build the two summary tables (cluster level, group level).

Consolidates the arm A findings into the minimum set of tables for the response,
recomputing everything from the artifacts rather than from notes. Includes the
two reproducibility floors, which are the only numbers that make the
cross-encoder figures interpretable:

  cluster floor   two MiniBatchKMeans seeds on the same uniform sample, per month
  group floor     the same two seeds carried through UMAP+HDBSCAN alignment, so
                  group identity is compared under nothing but a k-means restart

Inputs: 10f artifacts (cc/<tag>/cc_*.npz, assign/*.npz), the arm A group labels
and per-space drift tables from 10g/10h. If the seed-1 alignment is missing it
is computed here (~1 min) and cached.

Run from src/ (after 10f, 10g --stages armA, 10h --stages month,rsa,groups):
    python 10i_armA_summary.py --seed 0
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')

import argparse
import configparser
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

import matching
import ws3_common as wc
from matching import pair_scores

UMAP_KW = dict(n_neighbors=15, n_components=10, metric='euclidean',
               init='spectral', min_dist=0.1, spread=1.0)
HDBSCAN_KW = dict(min_cluster_size=3, min_samples=None,
                  cluster_selection_epsilon=0.0, max_cluster_size=100,
                  metric='euclidean')


def align(C, seed):
    import umap
    from sklearn.cluster import HDBSCAN
    Cu = umap.UMAP(**dict(UMAP_KW, random_state=seed)).fit_transform(C)
    return HDBSCAN(**HDBSCAN_KW).fit(Cu).labels_


def md(df, path, note=''):
    """Markdown table without pandas.to_markdown, which needs `tabulate` --
    not in the venv and not in the cm image's requirements.txt."""
    def fmt(v):
        if isinstance(v, float):
            return '' if pd.isna(v) else f'{v:.4f}'
        return '' if v is None else str(v)
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    with open(path, 'w') as f:
        if note:
            f.write(note.rstrip() + '\n\n')
        f.write('\n'.join(lines) + '\n')


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--models', type=str, default='minilm,mpnet,bge')
    p.add_argument('--k', type=int, default=wc.K_CLUSTERS)
    p.add_argument('--group_n', type=int, default=20000)
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--out_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()
    rev = os.path.join(g['save_path'], args.rev_subpath)
    tags = [t.strip() for t in args.models.split(',')]
    months = wc.ym_range('2006-01', '2022-12')
    T = args.out_path

    # ---------- cluster level ----------
    mo = pd.read_csv(os.path.join(T, 'R2.1_armA_month_agreement_draft.csv'))
    rsa = pd.read_csv(os.path.join(T, 'R2.1_armA_rsa_draft.csv'))
    rsa_m = rsa.groupby(['a', 'b']).mean(numeric_only=True)

    def rsa_get(a, b, col):
        for key in [(a, b), (b, a)]:
            if key in rsa_m.index:
                return rsa_m.loc[key, col]
        return np.nan

    rows = [dict(comparison='MiniLM seed 0 vs seed 1 (floor)',
                 purity=mo.pur_seed_minilm.mean(), ari=mo.ari_seed_minilm.mean(),
                 ami=np.nan, cka=np.nan, knn_overlap=np.nan,
                 layout_spearman=np.nan,
                 homogeneity=np.nan, completeness=np.nan),
            dict(comparison='published vs MiniLM-sample (subset effect)',
                 purity=mo.pur_pub_minilm.mean(), ari=mo.ari_pub_minilm.mean(),
                 ami=mo.ami_pub_minilm.mean(), cka=np.nan, knn_overlap=np.nan,
                 layout_spearman=np.nan,
                 homogeneity=rsa_get('published', 'minilm', 'homogeneity'),
                 completeness=rsa_get('published', 'minilm', 'completeness'))]
    for a, b in [('minilm', 'mpnet'), ('minilm', 'bge'), ('mpnet', 'bge')]:
        rows.append(dict(
            comparison=f'{a} vs {b} (model effect)',
            purity=mo[f'pur_{a}_{b}'].mean(), ari=mo[f'ari_{a}_{b}'].mean(),
            ami=mo[f'ami_{a}_{b}'].mean(),
            cka=rsa_get(a, b, 'cka'), knn_overlap=rsa_get(a, b, 'knn_overlap'),
            layout_spearman=rsa_get(a, b, 'layout_spearman'),
            homogeneity=rsa_get(a, b, 'homogeneity'),
            completeness=rsa_get(a, b, 'completeness')))
    cl = pd.DataFrame(rows)
    cl['pct_of_floor'] = cl.purity / cl.purity.iloc[0]
    cl.to_csv(os.path.join(T, 'R2.1_armA_cluster_level.csv'), index=False)
    md(cl, os.path.join(T, 'R2.1_armA_cluster_level.md'),
       'Arm A, cluster (month) level. Agreement between partitions of the same '
       'uniform sample. purity/ari/ami are means over 204 months; cka, '
       'knn_overlap and layout_spearman are means over 6 spread months and use '
       'no clustering or keywords. Floor = two k-means seeds on identical data.')
    print(cl.round(4).to_string(index=False))

    # ---------- group level ----------
    GL = {t: np.load(os.path.join(rev, 'align', f'armA_labels_{t}.npz'))['labels']
          for t in tags}
    fp = os.path.join(rev, 'align', 'armA_labels_minilm_seed1.npz')
    if os.path.exists(fp):
        GL['minilm_seed1'] = np.load(fp)['labels']
    else:
        C1 = np.vstack([np.load(os.path.join(rev, 'cc', 'minilm', f'cc_{m}.npz'),
                                allow_pickle=False)['armA_cc'][1]
                        for m in months]).astype(np.float64)
        GL['minilm_seed1'] = align(C1, args.seed)
        np.savez_compressed(fp, labels=GL['minilm_seed1'])
    rng = np.random.default_rng(args.seed)
    cols = {t: [] for t in list(tags) + ['minilm_seed1']}
    for mi, ym in enumerate(months):
        z = {t: np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                        allow_pickle=False)['armA_labels'] for t in tags}
        n = z[tags[0]].shape[1]
        idx = rng.choice(n, size=min(args.group_n, n), replace=False)
        for t in tags:
            cols[t].append(GL[t][mi * args.k + z[t][0][idx].astype(np.int64)])
        cols['minilm_seed1'].append(
            GL['minilm_seed1'][mi * args.k + z['minilm'][1][idx].astype(np.int64)])
    G = {t: np.concatenate(v) for t, v in cols.items()}

    grows = []
    for label, (a, b) in [('MiniLM seed 0 vs seed 1 (floor)', ('minilm', 'minilm_seed1')),
                          ('minilm vs mpnet (model effect)', ('minilm', 'mpnet')),
                          ('minilm vs bge (model effect)', ('minilm', 'bge')),
                          ('mpnet vs bge (model effect)', ('mpnet', 'bge'))]:
        grows.append(dict(comparison=label, **pair_scores(G[a], G[b])))
    gl = pd.DataFrame(grows)
    gl['pct_of_floor'] = gl.purity_1to1 / gl.purity_1to1.iloc[0]
    gl.to_csv(os.path.join(T, 'R2.1_armA_group_level.csv'), index=False)
    md(gl, os.path.join(T, 'R2.1_armA_group_level.md'),
       'Arm A, group level, compared at the comment level (every sampled comment '
       'belongs to one group per space) with comments in HDBSCAN noise on either '
       'side excluded. Floor = the same two k-means seeds carried through the '
       'alignment, so only a restart separates the two groupings.')
    print('\n' + gl.round(4).to_string(index=False))

    # structure + significance, with the chance baseline for correspondence
    st = pd.read_csv(os.path.join(T, 'R2.1_armA_group_structure_draft.csv'))
    dr = {t: pd.read_csv(os.path.join(T, f'R2.1_armA_groups_{t}_draft.csv'))
          for t in tags}
    srows = []
    for t in tags:
        d = dr[t]
        srows.append(dict(space=t,
                          n_groups=int(st.loc[st.space == t, 'n_groups'].iloc[0]),
                          noise_frac=float(st.loc[st.space == t, 'noise_frac'].iloc[0]),
                          median_span_months=float(
                              st.loc[st.space == t, 'median_span_months'].iloc[0]),
                          n_tested=len(d), median_T=float(d['T'].median()),
                          median_delta=float(d.delta.median()),
                          n_sig_q05=int((d.q_lam < 0.05).sum()),
                          rate_sig_q05=float((d.q_lam < 0.05).mean())))
    sg = pd.DataFrame(srows)
    base = tags[0]
    sigb = set(dr[base].coi[dr[base].q_lam < 0.05].astype(int))
    for t in tags[1:]:
        m = (G[base] >= 0) & (G[t] >= 0)
        A, B = G[base][m], G[t][m]
        qt = dr[t].set_index('coi')['q_lam']
        hit = tot = 0
        for gi in sigb:
            sel = B[A == gi]
            if not len(sel):
                continue
            vals, cnt = np.unique(sel, return_counts=True)
            bg = int(vals[cnt.argmax()])
            tot += 1
            if bg in qt.index and qt.loc[bg] < 0.05:
                hit += 1
        p0 = float((dr[t].q_lam < 0.05).mean())
        et = matching.enrichment_test(hit, tot, p0)
        sg.loc[sg.space == t, 'sig_corresponds_from_minilm'] = f'{hit}/{tot}'
        sg.loc[sg.space == t, 'corresponds_rate'] = et['rate']
        sg.loc[sg.space == t, 'chance_rate'] = et['chance_rate']
        sg.loc[sg.space == t, 'enrichment'] = et['enrichment']
        sg.loc[sg.space == t, 'binom_p'] = et['p']
    sg.to_csv(os.path.join(T, 'R2.1_armA_group_structure.csv'), index=False)
    md(sg, os.path.join(T, 'R2.1_armA_group_structure.md'),
       'Arm A group structure and drift results per space. Groups are re-derived '
       'per space, so only the RATE of significance is comparable, not the count '
       'of specific groups. The last columns ask whether a MiniLM significant '
       'group has a best-overlap counterpart that is also significant, against '
       'the chance rate in that space.')
    print('\n' + sg.round(4).to_string(index=False))
    print(f'\nwrote R2.1_armA_{{cluster_level,group_level,group_structure}}.'
          f'{{csv,md}} -> {T}')


if __name__ == '__main__':
    main()
