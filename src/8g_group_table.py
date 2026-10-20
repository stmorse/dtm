"""
WS1 / R1.1 step 8a: the per-group results table.

Joins the locked drift-test results with the step diagnostics, adds TF-IDF
labels, lifespan and path-shape columns, applies Benjamini-Hochberg FDR across
all tested groups, and writes

  R1.1_groups_all<tag>.csv        every tested group, every column, sorted by q
  R1.1_table1<tag>.{csv,md}       revised Table 1: the FDR-significant groups
  R1.1_table1_published<tag>.md   the topics named in the published Table 1 /
                                  Fig 4, with their corrected p and q

The all-groups CSV is the input to 8h_drift_map.py, so curation decisions
(which groups to highlight, which to drop as artifacts) are made once, on the
CSV, and the figure follows.

Run from src/ (needs /mnt/sciclone mounted):
    python 8g_group_table.py --seed 0
    python 8g_group_table.py --exclude 561,1002 --top 30
"""

import argparse
import configparser
import json
import os
import time

import numpy as np
import pandas as pd

import drift

# topics named in the published Table 1 and Fig 4: coi -> (name, where)
PUBLISHED_TOPICS = {
    290: ('conspiracy', 'Table 1 top'),
    771: ('oil', 'Table 1 top'),
    798: ('scientific', 'Table 1 top'),
    949: ('government', 'Table 1 top'),
    999: ('republican', 'Table 1 top'),
    510: ('woman', 'Table 1 bottom'),
    244: ('music', 'Table 1 bottom'),
    146: ('team', 'Table 1 bottom'),
    766: ('games', 'Table 1 bottom'),
    425: ('delicious', 'Table 1 bottom'),
    774: ('laughing', 'Fig 4b'),
    811: ('religion', 'Fig 4c'),
    1001: ('president', 'Fig 4a'),
}

ALL_COLS = ['kw', 'name', 'T', 'n_months', 'n_rows', 'first_ym', 'last_ym',
            'span_months', 'coverage', 'r', 'disp', 'diam', 'path_len',
            'straightness', 'max_step', 'lam', 'p_lam', 'q_lam',
            'd2', 'p_d2', 'q_d2', 'rho1', 'rho1_z', 'gap_rho', 'gap_n', 'shapiro_frac_lt05',
            'sig_q05', 'sig_q01', 'rank_q', 'published', 'kw_long']

TABLE1_COLS = ['rank_q', 'coi', 'kw', 'T', 'first_ym', 'last_ym',
               'disp', 'straightness', 'lam', 'p_lam', 'q_lam', 'p_d2', 'name']


def ym(period):
    """'YYYY-MM' for a period index (0 = 2006-01)."""
    return f'{2006 + period // 12}-{period % 12 + 1:02d}'


def parse_cois(s):
    return [] if not s else [int(x) for x in s.replace(' ', '').split(',') if x]


def fmt_p(x, floor=None):
    if pd.isna(x):
        return ''
    if floor is not None and x <= floor * (1 + 1e-9):
        return f'<{floor:.0e}'
    return f'{x:.1e}' if x < 0.001 else f'{x:.3f}'


def to_md(df, floats=3, p_cols=(), floor=None):
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    out = df.copy()
    for c in out.columns:
        if c in p_cols:
            out[c] = df[c].map(lambda x: fmt_p(x, floor))
        elif pd.api.types.is_float_dtype(df[c]):
            out[c] = df[c].map(lambda x: '' if pd.isna(x) else f'{x:.{floats}f}')
        else:
            out[c] = df[c].fillna('').astype(str)
    rows = ['| ' + ' | '.join(out.columns) + ' |',
            '|' + '|'.join(['---'] * len(out.columns)) + '|']
    rows += ['| ' + ' | '.join(r) + ' |' for r in out.astype(str).values]
    return '\n'.join(rows)


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='unused; kept for consistency')
    parser.add_argument('--results', type=str,
                        default='../output/ws1/8b/asis_group_half_r5_s0_B100k/groups.csv',
                        help='per-group CSV from 8b_drift_test.py (the locked run)')
    parser.add_argument('--diag', type=str,
                        default='../output/ws1/8f_diagnostics/groups.csv',
                        help='per-group CSV from 8f_diagnostics.py; skipped if absent')
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.05, help='q cutoff for Table 1')
    parser.add_argument('--top', type=int, default=0,
                        help='cap the Table 1 rows (0 = every group below alpha)')
    parser.add_argument('--exclude', type=str, default='',
                        help='coi to drop from Table 1 as artifacts (bots, spam, templates)')
    parser.add_argument('--kw_cache', type=str, default='../output/ws1/8g/keywords.pkl',
                        help='cache the TF-IDF keyword lists (the sshfs reads are slow)')
    parser.add_argument('--out_path', type=str, default='../output/ws1/8g')
    parser.add_argument('--tab_path', type=str, default='../refs/revision/tables')
    parser.add_argument('--tag', type=str, default='_draft')
    args = parser.parse_args()

    t0 = time.time()
    excl = parse_cois(args.exclude)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.tab_path, exist_ok=True)
    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)

    # provenance of the input run
    src_params = {}
    pf = os.path.join(os.path.dirname(args.results), 'params.json')
    if os.path.exists(pf):
        src_params = json.load(open(pf))
    n_perm = src_params.get('n_perm')
    floor = 1 / (n_perm + 1) if n_perm else None

    df = pd.read_csv(args.results).set_index('coi')

    # ---- BH-FDR over every tested group ---------------------------------
    for col in ['p_lam', 'p_d2']:
        if col in df:
            df['q' + col[1:]] = drift.bh_qvalues(df[col].values)

    # ---- labels and geometry --------------------------------------------
    C = drift.load_centroids(run_path, k=args.k)
    labels = drift.load_group_labels(align_path)
    keywords = load_keywords_cached(run_path, args.kw_cache)

    geo = []
    for coi in df.index:
        rows_k, per = drift.group_rows(labels, coi, k=args.k)
        X = drift.trajectory(C, rows_k)
        idx = np.where(labels == coi)[0]
        span = int(per[-1] - per[0]) + 1
        geo.append({
            'coi': coi,
            'kw': drift.group_label(keywords, idx, n=3),
            'kw_long': drift.group_label(keywords, idx, n=10),
            'first_ym': ym(int(per[0])), 'last_ym': ym(int(per[-1])),
            'span_months': span,
            'coverage': len(np.unique(per)) / span,
            'diam': drift.diameter(X),
            'path_len': drift.path_length(X),
            'straightness': drift.straightness(X),
            'max_step': drift.max_step(X),
        })
    df = df.join(pd.DataFrame(geo).set_index('coi'))

    # ---- diagnostics (optional) -----------------------------------------
    if args.diag and os.path.exists(args.diag):
        d = pd.read_csv(args.diag).set_index('coi')
        keep = [c for c in ['rho1', 'rho1_z', 'gap_rho', 'gap_n', 'shapiro_frac_lt05'] if c in d]
        df = df.join(d[keep])
    else:
        print(f'no diagnostics at {args.diag}: rho1 / Shapiro columns omitted')

    # ---- flags, names, ranks --------------------------------------------
    df['sig_q05'] = df.q_lam < 0.05
    df['sig_q01'] = df.q_lam < 0.01
    df['name'] = [PUBLISHED_TOPICS.get(c, ('', ''))[0] for c in df.index]
    df['published'] = [PUBLISHED_TOPICS.get(c, ('', ''))[1] for c in df.index]
    df = df.sort_values('q_lam', kind='stable')
    df['rank_q'] = np.arange(1, len(df) + 1)

    cols = [c for c in ALL_COLS if c in df]
    all_path = os.path.join(args.tab_path, f'R1.1_groups_all{args.tag}.csv')
    df[cols].to_csv(all_path)

    # ---- revised Table 1 -------------------------------------------------
    sig = df[df.q_lam < args.alpha]
    t1 = sig.drop(index=[c for c in excl if c in sig.index])
    n_excluded = len(sig) - len(t1)
    if args.top:
        t1 = t1.head(args.top)
    t1 = t1.reset_index()[[c for c in TABLE1_COLS if c in t1.reset_index().columns]]
    t1.to_csv(os.path.join(args.tab_path, f'R1.1_table1{args.tag}.csv'), index=False)

    header = (f'# R1.1 revised Table 1 - topic groups with BH-FDR q < {args.alpha}\n\n'
              f'Lambda on each group\'s own top-r_k principal axes, '
              f'r_k = min(5, floor((T-2)/2)); monthly centroids shuffled in time; '
              f'B = {n_perm:,} permutations; q from Benjamini-Hochberg over all '
              f'{len(df)} tested groups (T >= 5, label != -1).\n\n'
              f'{len(sig)} groups reach q < {args.alpha}'
              + (f'; {n_excluded} excluded as artifacts (coi {", ".join(map(str, excl))})'
                 if n_excluded else '') + f'; {len(t1)} shown.\n\n')
    with open(os.path.join(args.tab_path, f'R1.1_table1{args.tag}.md'), 'w') as f:
        f.write(header + to_md(t1, p_cols=('p_lam', 'q_lam', 'p_d2'), floor=floor) + '\n')

    # ---- what happened to the published topics ---------------------------
    pub = df[df.published != ''].copy()
    pub = pub.reset_index()[['name', 'published', 'coi', 'kw', 'T',
                             'disp', 'lam', 'p_lam', 'q_lam', 'rank_q']]
    pub = pub.sort_values(['published', 'q_lam'])
    pub.to_csv(os.path.join(args.tab_path, f'R1.1_table1_published{args.tag}.csv'), index=False)
    with open(os.path.join(args.tab_path, f'R1.1_table1_published{args.tag}.md'), 'w') as f:
        f.write('# R1.1 - the published Table 1 / Fig 4 topics under the corrected test\n\n'
                f'Same procedure and the same BH-FDR over all {len(df)} tested groups. '
                f'`rank_q` is the group\'s rank by q among those {len(df)}.\n\n')
        f.write(to_md(pub, p_cols=('p_lam', 'q_lam'), floor=floor) + '\n')

    summary = {
        'n_tested': len(df),
        'n_p05': int((df.p_lam < 0.05).sum()), 'n_p01': int((df.p_lam < 0.01).sum()),
        'n_q05': int((df.q_lam < 0.05).sum()), 'n_q01': int((df.q_lam < 0.01).sum()),
        'n_q10': int((df.q_lam < 0.10).sum()),
        'n_q05_d2': int((df.q_d2 < 0.05).sum()) if 'q_d2' in df else None,
        # BH's guarantee: at most alpha * (number rejected) false discoveries on average
        'expected_false_discoveries': round(args.alpha * float((df.q_lam < args.alpha).sum()), 2),
        'n_excluded': n_excluded,
        'published_topics_surviving': sorted(pub.loc[pub.q_lam < args.alpha, 'name'].tolist()),
        'straightness_median_sig': round(float(sig.straightness.median()), 3),
        'straightness_median_rest': round(float(df[df.q_lam >= args.alpha].straightness.median()), 3),
    }
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump(vars(args) | {'source_run': src_params, 'summary': summary,
                                'elapsed_s': round(time.time() - t0, 1)}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'\n{all_path}')
    print(to_md(t1.head(25), p_cols=('p_lam', 'q_lam', 'p_d2'), floor=floor))


def load_keywords_cached(run_path, cache):
    import pickle
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    kw = drift.load_keywords(run_path)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(kw, f)
    return kw


if __name__ == '__main__':
    main()
