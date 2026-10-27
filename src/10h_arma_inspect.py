"""
WS3 / R2.1 arm A detail: does month-level topic structure reproduce, and what do
the resulting topics look like?

Arm A only. Arm B1/B2 (inherited partitions / inherited groups) are in
10g_compare.py and are not touched here.

Stages (--stages):
  month   Per-month agreement between arm A partitions of the *same* uniform
          sample, with three comparisons that must be read together: published
          vs MiniLM-sample (subset effect), seed vs seed in one space (the
          within-model floor), and cross-model pairs (the model effect).
          Reports ARI, AMI and matched purity, and breaks them down by year, so
          an era or corpus-size trend is visible (links to R2.3).
  topics  For --inspect_months, match each space's arm A clusters to MiniLM's
          one-to-one by member overlap (Hungarian), then print the matched pairs
          with TF-IDF keywords from the retrieved text. This is the qualitative
          answer: are the three encoders finding recognisably the same topics?
  groups  Re-align each space's arm A centroids (saving the group labels, which
          10g does not keep), then report group-size structure and TF-IDF
          keywords for the largest and the highest-displacement groups.

Run from src/ (after 10f_centroids.py):
    python 10h_arma_inspect.py --seed 0 --stages month
    python 10h_arma_inspect.py --seed 0 --stages topics --inspect_months 2016-11
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')

import argparse
import configparser
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

import drift
import matching
import ws3_common as wc
from matching import matched_purity

MODELS = ['minilm', 'mpnet', 'bge']
UMAP_KW = dict(n_neighbors=15, n_components=10, metric='euclidean',
               init='spectral', min_dist=0.1, spread=1.0)
HDBSCAN_KW = dict(min_cluster_size=3, min_samples=None,
                  cluster_selection_epsilon=0.0, max_cluster_size=100,
                  metric='euclidean')


def keywords_for(docs, top_m=8, max_df=0.5):
    """Top TF-IDF terms per document (one document = one cluster's comments)."""
    vec = TfidfVectorizer(max_df=max_df, use_idf=True, smooth_idf=True,
                          min_df=2, stop_words='english')
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return [[] for _ in docs]
    names = vec.get_feature_names_out()
    out = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        out.append(list(names[np.argsort(row)[::-1][:top_m]]))
    return out


def read_month_text(rev, ym, rows_wanted=None):
    """{row: body} for a month (optionally restricted)."""
    want = None if rows_wanted is None else set(int(r) for r in rows_wanted)
    out = {}
    with gzip.open(os.path.join(rev, 'text', f'text_{ym}.jsonl.gz'), 'rt') as f:
        for line in f:
            d = json.loads(line)
            if want is None or d['row'] in want:
                out[d['row']] = d['body']
    return out


def align_units(C, seed, align_dim=10):
    import umap
    from sklearn.cluster import HDBSCAN
    Cu = umap.UMAP(**dict(UMAP_KW, n_components=align_dim,
                          random_state=seed)).fit_transform(C)
    return HDBSCAN(**HDBSCAN_KW).fit(Cu).labels_


# ------------------------------------------------------------------ stages

def stage_month(rev, months, tags, args, out):
    recs = []
    for ym in months:
        a = np.load(os.path.join(rev, 'assign', f'assign_{ym}.npz'),
                    allow_pickle=False)
        uni = (a['flags'] & 1) > 0
        pub = a['assign'][uni].astype(np.int64)
        lab = {t: np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                          allow_pickle=False)['armA_labels'].astype(np.int64)
               for t in tags}
        rec = {'month': ym, 'year': int(ym[:4]), 'n': int(uni.sum())}
        if 'minilm' in lab:
            rec['ari_pub_minilm'] = ari(pub, lab['minilm'][0])
            rec['ami_pub_minilm'] = ami(pub, lab['minilm'][0])
            rec['pur_pub_minilm'] = matched_purity(pub, lab['minilm'][0], args.k)[0]
        for t in tags:
            if lab[t].shape[0] > 1:
                rec[f'ari_seed_{t}'] = ari(lab[t][0], lab[t][1])
                rec[f'pur_seed_{t}'] = matched_purity(lab[t][0], lab[t][1], args.k)[0]
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                rec[f'ari_{t1}_{t2}'] = ari(lab[t1][0], lab[t2][0])
                rec[f'ami_{t1}_{t2}'] = ami(lab[t1][0], lab[t2][0])
                rec[f'pur_{t1}_{t2}'] = matched_purity(lab[t1][0], lab[t2][0],
                                                       args.k)[0]
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armA_month_agreement_draft.csv'), index=False)
    key = [c for c in df.columns if c.startswith(('ari_', 'ami_', 'pur_'))]
    print('arm A, month-level agreement over %d months (mean):' % len(df))
    print(df[key].mean().round(4).to_string())
    print('\nby year (matched purity, the readable one):')
    pur = [c for c in df.columns if c.startswith('pur_')]
    byyear = df.groupby('year')[pur].mean()
    print(byyear.round(3).to_string())
    byyear.to_csv(os.path.join(out, 'R2.1_armA_month_agreement_byyear_draft.csv'))
    print('\ncorrelation of agreement with month size (log n of the month):')
    tot = wc.load_totals(args.save_path)
    ln = np.log10([tot[m] for m in df.month])
    from scipy.stats import spearmanr
    for c in pur:
        print(f'  {c}: spearman {spearmanr(ln, df[c]).statistic:+.3f}')
    return df


def stage_topics(rev, months, tags, args, out):
    k = args.k
    for ym in [m.strip() for m in args.inspect_months.split(',') if m.strip()]:
        print(f'\n===== {ym} =====')
        a = np.load(os.path.join(rev, 'assign', f'assign_{ym}.npz'),
                    allow_pickle=False)
        uni = (a['flags'] & 1) > 0
        rows = a['rows'][uni]
        pub = a['assign'][uni].astype(np.int64)
        lab = {t: np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                          allow_pickle=False)['armA_labels'][0].astype(np.int64)
               for t in tags}
        txt = read_month_text(rev, ym, rows)
        bodies = np.array([txt.get(int(r), '') for r in rows], dtype=object)

        kw = {}
        for name, lb in [('published', pub)] + [(t, lab[t]) for t in tags]:
            docs = []
            for c in range(k):
                sel = np.flatnonzero(lb == c)[:args.max_docs]
                docs.append(' '.join(bodies[sel]))
            kw[name] = keywords_for(docs, args.top_m)

        base = 'minilm'
        for t in [x for x in tags if x != base] + ['published']:
            lb_b = lab[base]
            lb_t = pub if t == 'published' else lab[t]
            pur, M = matched_purity(lb_b, lb_t, k)
            r, c = linear_sum_assignment(-M)
            print(f'\n-- {base} vs {t}: matched purity {pur:.3f} '
                  f'(ARI {ari(lb_b, lb_t):.3f}) --')
            order = np.argsort(-M[r, c])
            print(f'{"ov%":>5}  {base:<42} {t}')
            for j in order[:args.show]:
                i1, i2 = r[j], c[j]
                ov = M[i1, i2] / max(1, (lb_b == i1).sum())
                print(f'{ov*100:5.0f}  {" ".join(kw[base][i1][:5]):<42} '
                      f'{" ".join(kw[t][i2][:5])}')
        with open(os.path.join(out, f'R2.1_armA_topics_{ym}_draft.json'), 'w') as f:
            json.dump(kw, f, indent=2)


def stage_groups(rev, months, tags, args, out):
    k = args.k
    recs = []
    for t in tags:
        C = np.vstack([np.load(os.path.join(rev, 'cc', t, f'cc_{m}.npz'),
                               allow_pickle=False)['armA_cc'][0]
                       for m in months]).astype(np.float64)
        lp = os.path.join(rev, 'align', f'armA_labels_{t}.npz')
        if os.path.exists(lp) and not args.overwrite:
            lab = np.load(lp)['labels']
        else:
            lab = align_units(C, args.seed, args.align_dim)
            np.savez_compressed(lp, labels=lab)
        sizes = np.array([(lab == gi).sum()
                          for gi in np.unique(lab) if gi >= 0])
        tested = drift.tested_groups(lab, min_size=5)
        spans = []
        for gi in tested:
            per = np.unique(np.flatnonzero(lab == gi) // k)
            spans.append(per[-1] - per[0] + 1)
        recs.append({'space': t, 'n_groups': len(sizes),
                     'noise_frac': float((lab == -1).mean()),
                     'median_size': float(np.median(sizes)),
                     'max_size': int(sizes.max()),
                     'n_tested': len(tested),
                     'median_span_months': float(np.median(spans)),
                     'frac_span_gt_60': float(np.mean(np.array(spans) > 60))})
        print(f'  {t}: {recs[-1]}')
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armA_group_structure_draft.csv'), index=False)
    print('\narm A group structure:')
    print(df.round(4).to_string(index=False))
    return df


def linear_cka(X, Y):
    """
    Linear CKA between two representations of the same points. Invariant to
    rotation, isotropic scaling and translation, so it compares the *geometry*
    the two encoders impose, with no clustering and no keywords. 1 = same up to
    those invariances. Failure mode: it is a global, second-order summary, so it
    can be high while local neighbourhoods differ, and it is sensitive to the
    few dominant directions of variance.
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy))


def knn_overlap(X, Y, kn=10):
    """
    Mean fraction of each point's kn nearest neighbours that are shared between
    two spaces. Local, comment-level, and independent of any clustering: it asks
    whether the two encoders put the same comments next to each other. Failure
    mode: depends on kn, and in dense regions many neighbours are near-ties, so
    moderate overlap can understate agreement.
    """
    def nn(Z):
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, kn, axis=1)[:, :kn]
    A, B = nn(X), nn(Y)
    return float(np.mean([len(set(a) & set(b)) / kn for a, b in zip(A, B)]))


def stage_rsa(rev, months, tags, args, out):
    """Correspondence without keywords: CKA, kNN overlap, inter-cluster layout,
    and homogeneity/completeness (which separates merges from splits)."""
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv
    from scipy.stats import spearmanr
    sel_months = [m.strip() for m in args.rsa_months.split(',') if m.strip()]
    rng = np.random.default_rng(args.seed)
    recs = []
    for ym in sel_months:
        a = np.load(os.path.join(rev, 'assign', f'assign_{ym}.npz'),
                    allow_pickle=False)
        uni = (a['flags'] & 1) > 0
        pub = a['assign'][uni].astype(np.int64)
        X, lab, cc = {}, {}, {}
        for t in tags:
            e = np.load(os.path.join(rev, 'emb', t, f'emb_{ym}.npz'),
                        allow_pickle=False)
            X[t] = e['X'][uni].astype(np.float32)
            z = np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                        allow_pickle=False)
            lab[t] = z['armA_labels'][0].astype(np.int64)
            cc[t] = z['armA_cc'][0].astype(np.float64)
        n = len(pub)
        idx = rng.choice(n, size=min(args.rsa_n, n), replace=False)
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                h, c, v = hcv(lab[t1], lab[t2])
                # inter-cluster layout: match clusters by member overlap, then
                # compare the two 50x50 centroid-distance matrices
                _, M = matched_purity(lab[t1], lab[t2], args.k)
                r, cm = linear_sum_assignment(-M)
                D1 = np.linalg.norm(cc[t1][r][:, None, :] - cc[t1][r][None, :, :], axis=-1)
                D2 = np.linalg.norm(cc[t2][cm][:, None, :] - cc[t2][cm][None, :, :], axis=-1)
                iu = np.triu_indices(len(r), 1)
                recs.append({
                    'month': ym, 'a': t1, 'b': t2,
                    'cka': linear_cka(X[t1][idx], X[t2][idx]),
                    'knn_overlap': knn_overlap(X[t1][idx], X[t2][idx], args.knn),
                    'layout_spearman': spearmanr(D1[iu], D2[iu]).statistic,
                    'homogeneity': h, 'completeness': c, 'v_measure': v})
        # same-encoder reference: published partition vs the MiniLM refit
        if 'minilm' in lab:
            h, c, v = hcv(pub, lab['minilm'])
            recs.append({'month': ym, 'a': 'published', 'b': 'minilm',
                         'cka': np.nan, 'knn_overlap': np.nan,
                         'layout_spearman': np.nan,
                         'homogeneity': h, 'completeness': c, 'v_measure': v})
        print(f'  {ym} done', flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armA_rsa_draft.csv'), index=False)
    print('\ncorrespondence without keywords (mean over %d months):' % len(sel_months))
    print(df.groupby(['a', 'b'])[['cka', 'knn_overlap', 'layout_spearman',
                                  'homogeneity', 'completeness', 'v_measure']]
          .mean().round(4).to_string())
    return df


def stage_grouplevel(rev, months, tags, args, out):
    """
    Group-level comparison, done at the comment level so the spaces are
    comparable. Every sampled comment maps to a (month, cluster) unit in each
    space, hence to one arm A group (or noise), so the same agreement measures
    apply to groups as to clusters.
    """
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv
    from scipy.stats import spearmanr
    k = args.k
    allm = [f'{y}-{m:02d}' for y in range(2006, 2023) for m in range(1, 13)]
    gl = {t: np.load(os.path.join(rev, 'align', f'armA_labels_{t}.npz'))['labels']
          for t in tags}
    rng = np.random.default_rng(args.seed)
    cols = {t: [] for t in tags}
    for ym in months:
        per = allm.index(ym)
        z0 = np.load(os.path.join(rev, 'cc', tags[0], f'cc_{ym}.npz'),
                     allow_pickle=False)
        n = z0['armA_labels'].shape[1]
        idx = rng.choice(n, size=min(args.group_n, n), replace=False)
        for t in tags:
            cl = np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                         allow_pickle=False)['armA_labels'][0][idx].astype(np.int64)
            cols[t].append(gl[t][per * k + cl])
    G = {t: np.concatenate(cols[t]) for t in tags}
    keep = np.ones(len(G[tags[0]]), dtype=bool)
    for t in tags:
        keep &= G[t] >= 0                      # drop comments in noise anywhere
    print(f'  pooled {len(keep):,} comments; {keep.sum():,} grouped in all spaces '
          f'({keep.mean():.1%})')
    recs = []
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1:]:
            h, c, v = hcv(G[t1][keep], G[t2][keep])
            recs.append({'a': t1, 'b': t2, 'ari': ari(G[t1][keep], G[t2][keep]),
                         'ami': ami(G[t1][keep], G[t2][keep]),
                         'homogeneity': h, 'completeness': c, 'v_measure': v})
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, 'R2.1_armA_group_agreement_draft.csv'), index=False)
    print('\ngroup-level agreement, comment level, noise excluded:')
    print(df.round(4).to_string(index=False))

    # match groups across spaces by shared comments, then compare drift results
    base = tags[0]
    dfs = {t: pd.read_csv(os.path.join(out, f'R2.1_armA_groups_{t}_draft.csv'))
           for t in tags}
    for t in tags[1:]:
        # matching.match_many_to_one filters to non-noise in both internally;
        # `keep` is already stricter (non-noise in every space), so that is a
        # no-op here and frac_to_noise is 0 by construction. WS2 should pass
        # unfiltered labels to get a meaningful frac_to_noise.
        mt = matching.match_many_to_one(G[base][keep], G[t][keep])
        mt = mt.merge(dfs[base][['coi', 'delta', 'q_lam', 'T']]
                      .rename(columns={'coi': 'g_a', 'delta': 'delta_base',
                                       'q_lam': 'q_base', 'T': 'T_base'}),
                      on='g_a')
        mt = mt.merge(dfs[t][['coi', 'delta', 'q_lam', 'T']]
                      .rename(columns={'coi': 'g_b', 'delta': 'delta_other',
                                       'q_lam': 'q_other', 'T': 'T_other'}),
                      on='g_b')
        strong = mt[mt.overlap >= args.min_overlap]
        sb = set(strong.g_a[strong.q_base < 0.05])
        so = set(strong.g_a[strong.q_other < 0.05])
        print(f'\n  {base} vs {t}: {len(mt)} matched groups, '
              f'{len(strong)} with overlap >= {args.min_overlap}')
        print(f'    median matching overlap: {mt.overlap.median():.3f}')
        print(f'    delta spearman (strong matches): '
              f'{spearmanr(strong.delta_base, strong.delta_other).statistic:.3f}')
        print(f'    T spearman: {spearmanr(strong.T_base, strong.T_other).statistic:.3f}')
        print(f'    significant q<0.05: {len(sb)} in {base}, {len(so)} in {t}; '
              f'overlap {len(sb & so)} '
              f'(containment of smaller: '
              f'{matching.containment(sb, so):.3f})')
        mt.to_csv(os.path.join(out, f'R2.1_armA_group_match_{base}_{t}_draft.csv'),
                  index=False)
    return df


def stage_groupkw(rev, months, tags, args, out):
    """
    Group-level qualitative view: TF-IDF keywords per arm A group in each space,
    then the correspondence among groups that pass q<0.05. Keywords are read-only
    labels; all matching is by shared comments.
    """
    k = args.k
    allm = [f'{y}-{m:02d}' for y in range(2006, 2023) for m in range(1, 13)]
    GL = {t: np.load(os.path.join(rev, 'align', f'armA_labels_{t}.npz'))['labels']
          for t in tags}
    pool = {t: {} for t in tags}       # group -> reservoir of bodies
    seen = {t: {} for t in tags}
    gmap = {t: [] for t in tags}       # per-comment group label, pooled
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for mi, ym in enumerate(months):
        per = allm.index(ym)
        z = {t: np.load(os.path.join(rev, 'cc', t, f'cc_{ym}.npz'),
                        allow_pickle=False) for t in tags}
        rows = z[tags[0]]['armA_rows']
        txt = read_month_text(rev, ym, rows)
        bodies = [txt.get(int(r), '') for r in rows]
        sub = rng.choice(len(rows), size=min(args.group_n, len(rows)), replace=False)
        for t in tags:
            grp = GL[t][per * k + z[t]['armA_labels'][0].astype(np.int64)]
            gmap[t].append(grp[sub])
            for i, gi in enumerate(grp):
                if gi < 0:
                    continue
                gi = int(gi)
                seen[t][gi] = seen[t].get(gi, 0) + 1
                res = pool[t].setdefault(gi, [])
                if len(res) < args.kw_max:       # reservoir, so months mix evenly
                    res.append(bodies[i])
                else:
                    j = rng.integers(0, seen[t][gi])
                    if j < args.kw_max:
                        res[j] = bodies[i]
        if mi % 40 == 0:
            print(f'  {ym} ({time.time()-t0:.0f}s)', flush=True)
    G = {t: np.concatenate(gmap[t]) for t in tags}

    rec = []
    kw = {}
    for t in tags:
        gids = sorted(pool[t])
        docs = [' '.join(pool[t][gi]) for gi in gids]
        kws = keywords_for(docs, args.top_m)
        kw[t] = dict(zip(gids, kws))
        dft = pd.read_csv(os.path.join(out, f'R2.1_armA_groups_{t}_draft.csv')) \
            .set_index('coi')
        for gi in gids:
            r = {'space': t, 'group': gi, 'n_comments': seen[t][gi],
                 'keywords': ' '.join(kws[gi])}
            if gi in dft.index:
                for c in ['T', 'delta', 'p_lam', 'q_lam']:
                    r[c] = dft.loc[gi, c]
            rec.append(r)
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(out, 'R2.1_armA_group_keywords_draft.csv'), index=False)

    for t in tags:
        sig = df[(df.space == t) & (df.q_lam < 0.05)].sort_values('q_lam')
        print(f'\n=== {t}: {len(sig)} groups at q<0.05, top {args.show} by q ===')
        for _, r in sig.head(args.show).iterrows():
            print(f'  g{int(r.group):<5} T={int(r["T"]):>3} d={r.delta:.3f} '
                  f'q={r.q_lam:.4f}  {r.keywords}')

    base = tags[0]
    sigb = set(df[(df.space == base) & (df.q_lam < 0.05)].group.astype(int))
    print(f'\n=== correspondence of {base} q<0.05 groups in the other spaces ===')
    print('(match is by shared comments; "sig" = that counterpart also q<0.05)')
    for t in tags[1:]:
        m = (G[base] >= 0) & (G[t] >= 0)
        A, B = G[base][m], G[t][m]
        qt = df[df.space == t].set_index('group')
        n_sig_counterpart = 0
        shown = 0
        print(f'\n-- {base} -> {t} --')
        for gi in sorted(sigb):
            sel = B[A == gi]
            if not len(sel):
                continue
            vals, cnt = np.unique(sel, return_counts=True)
            bg = int(vals[cnt.argmax()]); ov = cnt.max() / len(sel)
            q = qt.loc[bg, 'q_lam'] if bg in qt.index else np.nan
            if not np.isnan(q) and q < 0.05:
                n_sig_counterpart += 1
            if shown < args.show:
                print(f'  g{gi:<5} ov {ov:.2f} -> g{bg:<5} q={q if np.isnan(q) else round(q,4)}'
                      f'  | {base}: {kw[base][gi][:4]}')
                print(f'                                      | {t}: {kw[t].get(bg, [])[:4]}')
                shown += 1
        print(f'  of {len(sigb)} {base} significant groups, {n_sig_counterpart} have a '
              f'best-overlap counterpart in {t} that is also q<0.05 '
              f'({n_sig_counterpart/max(1,len(sigb)):.1%})')
    return df


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stages', type=str, default='month')
    p.add_argument('--models', type=str, default=','.join(MODELS))
    p.add_argument('--months', type=str, default='')
    p.add_argument('--inspect_months', type=str, default='2016-11')
    p.add_argument('--k', type=int, default=wc.K_CLUSTERS)
    p.add_argument('--top_m', type=int, default=8)
    p.add_argument('--show', type=int, default=14)
    p.add_argument('--max_docs', type=int, default=1500)
    p.add_argument('--align_dim', type=int, default=10)
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--out_path', type=str, default='../refs/revision/tables')
    p.add_argument('--rsa_months', type=str,
                   default='2008-06,2012-06,2016-06,2016-11,2020-04,2022-06')
    p.add_argument('--rsa_n', type=int, default=3000)
    p.add_argument('--knn', type=int, default=10)
    p.add_argument('--group_n', type=int, default=20000)
    p.add_argument('--min_overlap', type=float, default=0.5)
    p.add_argument('--kw_max', type=int, default=40,
                   help='comments kept per group for the keyword view')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    args.save_path = g['save_path']
    rev = os.path.join(g['save_path'], args.rev_subpath)
    os.makedirs(os.path.join(rev, 'align'), exist_ok=True)
    os.makedirs(args.out_path, exist_ok=True)
    tags = [t.strip() for t in args.models.split(',') if t.strip()]
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or sorted(f[len('cc_'):-len('.npz')] for f in
                        os.listdir(os.path.join(rev, 'cc', 'minilm'))
                        if f.startswith('cc_')))
    t0 = time.time()
    fns = {'month': stage_month, 'topics': stage_topics, 'groups': stage_groups,
           'rsa': stage_rsa, 'grouplevel': stage_grouplevel,
           'groupkw': stage_groupkw}
    for s in [x.strip() for x in args.stages.split(',') if x.strip()]:
        print(f'\n=== {s} ===')
        fns[s](rev, months, tags, args, args.out_path)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
