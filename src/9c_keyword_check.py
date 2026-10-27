"""
WS2 / R2.4: qualitative check of re-derived groupings, WS3's method.

Groups are matched by shared (month, cluster) units only; TF-IDF keywords
(the published per-cluster keywords) are read afterwards as labels, never as
the matching key.

Matched groups share units, so their labels agree partly by construction. Two
comparisons avoid that circularity and carry the evidence:
  newcomers   units the matched variant group holds that the published group
              did not -- are they the same topic?
  emigrants   the other (non-noise) variant groups that took some of the
              published group's units, labelled from their OTHER units -- did
              the group fragment into same-topic pieces, or scatter?
Each is a Jaccard between top-n keyword sets, reported against a null that
pairs each published group with a random variant group.

Run from src/ after 9b_align_grid.py:
    python 9c_keyword_check.py --seed 0
"""

import argparse
import configparser
import json
import os
import pickle

import numpy as np
import pandas as pd

import drift
import matching as mt

# groups named in the published Table 1 / Fig 4 (8g_group_table.PUBLISHED_TOPICS)
# and in WS1's revised headline examples
NAMED = [1001, 949, 999, 290, 771, 798, 510, 244, 146, 766, 425, 774, 811,
         1002, 983, 821, 856, 991, 221]


def top_words(kw, idx, n):
    return set(drift.group_label(kw, idx, n=n).split())


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else np.nan


def compare(pub, lab, kw, tested, n, min_units, rng):
    m = mt.match_many_to_one(pub, lab).set_index('g_a')
    gb_all = np.unique(lab[lab >= 0])
    recs = []
    for g in tested:
        u = np.flatnonzero(pub == g)
        w_pub = top_words(kw, u, n)
        r = {'g_pub': g, 'n_units': len(u), 'label_pub': drift.group_label(kw, u, n=5)}
        if g not in m.index or pd.isna(m.loc[g, 'g_b']):
            r.update(frac_to_noise=1.0)
            recs.append(r)
            continue
        gb = int(m.loc[g, 'g_b'])
        v = np.flatnonzero(lab == gb)
        new = np.setdiff1d(v, u)
        dest = lab[u]
        others = [d for d in np.unique(dest[dest >= 0]) if d != gb]
        r.update(g_var=gb, overlap=m.loc[g, 'overlap'],
                 frac_to_noise=m.loc[g, 'frac_to_noise'], n_dest=1 + len(others),
                 label_var=drift.group_label(kw, v, n=5),
                 j_matched=jacc(w_pub, top_words(kw, v, n)), n_new=len(new))
        if len(new) >= min_units:
            r['label_new'] = drift.group_label(kw, new, n=5)
            r['j_new'] = jacc(w_pub, top_words(kw, new, n))
        js, labs = [], []
        for d in others:
            rest = np.setdiff1d(np.flatnonzero(lab == d), u)
            if len(rest) >= min_units:
                js.append(jacc(w_pub, top_words(kw, rest, n)))
                labs.append(drift.group_label(kw, rest, n=3))
        if js:
            r['j_emigrant_median'] = float(np.median(js))
            r['labels_emigrant'] = ' | '.join(labs[:4])
        # null: a random variant group, labelled from units outside g
        d = rng.choice(gb_all)
        rest = np.setdiff1d(np.flatnonzero(lab == d), u)
        if len(rest) >= min_units:
            r['j_null'] = jacc(w_pub, top_words(kw, rest, n))
        recs.append(r)
    return pd.DataFrame(recs)


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variants', type=str,
                   default='published_s0,n_neighbors=50_s0,min_cluster_size=10_s0,no_umap_pca50_s0',
                   help='label files from 9b (output/ws2/9b/labels/<name>.npz)')
    p.add_argument('--n_words', type=int, default=10)
    p.add_argument('--min_units', type=int, default=2)
    p.add_argument('--n_random', type=int, default=15,
                   help='random tested groups shown alongside the named ones')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--align_subpath', type=str, default='align/max_100')
    p.add_argument('--grid_path', type=str, default='../output/ws2/9b')
    p.add_argument('--kw_cache', type=str, default='../output/ws2/keywords.pkl')
    p.add_argument('--out_path', type=str, default='../output/ws2/9c')
    p.add_argument('--tab_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump({**vars(args), 'save_path': g['save_path']}, f, indent=1)

    run_path = os.path.join(g['save_path'], args.run_subpath)
    pub = drift.load_group_labels(os.path.join(run_path, args.align_subpath))
    if os.path.exists(args.kw_cache):
        kw = pickle.load(open(args.kw_cache, 'rb'))
    else:
        kw = drift.load_keywords(run_path)
        pickle.dump(kw, open(args.kw_cache, 'wb'))
    tested = drift.tested_groups(pub)

    rng = np.random.default_rng(args.seed)
    show = [c for c in NAMED if c in tested]
    show += sorted(rng.choice([c for c in tested if c not in show],
                              size=args.n_random, replace=False).tolist())
    summ, frames = [], []
    for name in args.variants.split(','):
        lab = np.load(os.path.join(args.grid_path, 'labels', f'{name}.npz'))['labels']
        df = compare(pub, lab, kw, tested, args.n_words, args.min_units,
                     np.random.default_rng([args.seed, len(frames)]))
        df.insert(0, 'variant', name)
        frames.append(df)
        s = {'variant': name, 'n_pub_tested': len(df)}
        for c in ('overlap', 'frac_to_noise', 'n_dest', 'j_matched', 'j_new',
                  'j_emigrant_median', 'j_null'):
            s[c + '_median'] = float(df[c].median()) if c in df else np.nan
            s[c + '_n'] = int(df[c].notna().sum()) if c in df else 0
            if c.startswith('j_') and c in df:
                s[c + '_mean'] = float(df[c].mean())
                s[c + '_frac_pos'] = float((df[c].dropna() > 0).mean())
        summ.append(s)
        print(f'\n{name}:', json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                         for k, v in s.items()}))

    allg = pd.concat(frames, ignore_index=True)
    allg.to_csv(os.path.join(args.out_path, 'keyword_check_all.csv'), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(args.tab_path, 'R2.4_keyword_summary_draft.csv'),
                              index=False)
    ex = allg[allg.g_pub.isin(show)].copy()
    ex['shown_as'] = np.where(ex.g_pub.isin(NAMED), 'named', 'random')
    cols = ['variant', 'shown_as', 'g_pub', 'n_units', 'label_pub', 'overlap',
            'frac_to_noise', 'n_dest', 'label_var', 'label_new', 'labels_emigrant']
    ex[[c for c in cols if c in ex]].to_csv(
        os.path.join(args.tab_path, 'R2.4_keyword_examples_draft.csv'), index=False)
    print(f'\n-> {args.tab_path}/R2.4_keyword_{{summary,examples}}_draft.csv')


if __name__ == '__main__':
    main()
