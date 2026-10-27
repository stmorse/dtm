"""
WS3 / R2.1: qualitative inspection of the arm B2 groupings.

Arm B2 re-derives the grouping over the *identical* 10,200 (month, cluster)
units, so the published TF-IDF keyword list applies unchanged to every space's
groups. That makes the keyword view far cleaner here than in arm A, where the
units themselves differ and keywords had to be re-derived from re-sampled text:
nothing is re-estimated, and both sides of every comparison are described by the
same 20 terms per unit.

The question the low ARI leaves open is which of two things it means:
  (a) both groupings are semantically coherent and simply carve the same space
      at different boundaries, or
  (b) one of them is not tracking topic at all.
Stage `coherence` separates these. Stage `match` then asks whether the groups
that do not correspond by membership nonetheless correspond by content, and
stage `fate` shows, for named published groups, exactly where their members go.

Jaccard is safe in this script and only here: every unit carries exactly 20
TF-IDF terms, so the set-size confound that rules it out for the significant-set
comparison does not arise.

Run from src/ (after 10g_compare.py --stages armB2):
    python 10j_armB_inspect.py --seed 0
    python 10j_armB_inspect.py --seed 0 --stages fate --groups 1002,983,221
"""

import argparse
import configparser
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import drift
import ws3_common as wc

MODELS = ['minilm', 'mpnet', 'bge']
ALL_STAGES = ['coherence', 'match', 'fate']
SPACES = ['published', 'published_realigned'] + MODELS


def load_groupings(rev, save_path, run_subpath, align_subpath, tags):
    """{name: labels over the 10,200 units}. Published first, then the rungs."""
    out = {'published': drift.load_group_labels(
        os.path.join(save_path, run_subpath, align_subpath))}
    for name, fn in [('published_realigned', 'armB2_labels_control.npz')] + \
                    [(t, f'armB2_labels_{t}.npz') for t in tags]:
        p = os.path.join(rev, 'align', fn)
        if os.path.exists(p):
            lab = np.load(p, allow_pickle=False)['labels']
            if len(lab) != len(out['published']):
                raise RuntimeError(
                    f'{fn} has {len(lab):,} units, the published grouping has '
                    f'{len(out["published"]):,}. This is a truncated run\'s '
                    f'labels; re-run 10g_compare.py --stages armB2 on all months.')
            out[name] = lab
        else:
            print(f'  (no {fn}; run 10g_compare.py --stages armB2 first)')
    return out


def kw_sets(kw):
    """One frozenset of terms per unit, so overlap is a set operation."""
    return [frozenset(map(str, k)) for k in kw]


def coherence(sets, idx, rng=None, max_pairs=300):
    """
    Mean pairwise Jaccard of the member keyword sets of one group.

    Every unit has exactly 20 terms, so Jaccard is unconfounded here. Large
    groups are subsampled to max_pairs members, which only affects the variance.
    """
    if len(idx) < 2:
        return np.nan
    if len(idx) > max_pairs and rng is not None:
        idx = rng.choice(idx, size=max_pairs, replace=False)
    tot, n = 0.0, 0
    for i in range(len(idx)):
        a = sets[idx[i]]
        for j in range(i + 1, len(idx)):
            b = sets[idx[j]]
            tot += len(a & b) / len(a | b)
            n += 1
    return tot / n


def grouping_coherence(sets, labels, rng, n_null=5, min_size=3):
    """
    Per-group coherence, plus a size-matched null from shuffling the labels of
    the non-noise units. The null is the comparison that matters: a grouping
    with low ARI against another is still meaningful if its groups are far more
    keyword-coherent than a random partition of the same sizes.
    """
    keep = labels >= 0
    rows = np.flatnonzero(keep)
    obs = {}
    for c in np.unique(labels[keep]):
        idx = np.flatnonzero(labels == c)
        if len(idx) >= min_size:
            obs[int(c)] = coherence(sets, idx, rng)
    null = []
    for _ in range(n_null):
        perm = labels.copy()
        perm[rows] = labels[rng.permutation(rows)]
        vals = [coherence(sets, np.flatnonzero(perm == c), rng)
                for c in np.unique(perm[keep])
                if (perm == c).sum() >= min_size]
        null.append(np.nanmedian(vals))
    return obs, float(np.mean(null))


def group_profile(kw, idx, n=20):
    """A group's own keyword profile: the n most common member terms."""
    cnt = Counter(str(w) for i in idx for w in kw[i])
    return [w for w, _ in cnt.most_common(n)]


def match_groupings(lab_a, lab_b, kw, min_size=3, top=None):
    """
    Match a's groups to b's by shared units (Hungarian on the overlap matrix),
    then score each matched pair two ways: by membership containment and by
    keyword-profile overlap. The gap between the two is the whole point - a pair
    can share few units and still describe the same thing.
    """
    ca = [int(c) for c in np.unique(lab_a[lab_a >= 0])
          if (lab_a == c).sum() >= min_size]
    cb = [int(c) for c in np.unique(lab_b[lab_b >= 0])
          if (lab_b == c).sum() >= min_size]
    if top:                                   # keep the largest, for speed
        ca = sorted(ca, key=lambda c: -(lab_a == c).sum())[:top]
        cb = sorted(cb, key=lambda c: -(lab_b == c).sum())[:top]
    ia = {c: np.flatnonzero(lab_a == c) for c in ca}
    ib = {c: np.flatnonzero(lab_b == c) for c in cb}
    sa = {c: set(v.tolist()) for c, v in ia.items()}
    sb = {c: set(v.tolist()) for c, v in ib.items()}
    O = np.zeros((len(ca), len(cb)))
    for i, c in enumerate(ca):
        for j, e in enumerate(cb):
            O[i, j] = len(sa[c] & sb[e])
    r, cc = linear_sum_assignment(-O)
    pa = {c: set(group_profile(kw, ia[c])) for c in ca}
    pb = {c: set(group_profile(kw, ib[c])) for c in cb}
    rows = []
    for i, j in zip(r, cc):
        a, b = ca[i], cb[j]
        shared = O[i, j]
        rows.append({
            'a': a, 'b': b, 'n_a': len(sa[a]), 'n_b': len(sb[b]),
            'shared_units': int(shared),
            'containment_a': shared / len(sa[a]),
            'kw_overlap': len(pa[a] & pb[b]) / len(pa[a] | pb[b]),
        })
    df = pd.DataFrame(rows)
    # the null for kw_overlap: the same statistic over *mismatched* pairs
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cc))
    null = [len(pa[ca[r[i]]] & pb[cb[cc[perm[i]]]]) /
            len(pa[ca[r[i]]] | pb[cb[cc[perm[i]]]]) for i in range(len(r))]
    return df, float(np.median(null))


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stages', type=str, default=','.join(ALL_STAGES))
    p.add_argument('--models', type=str, default=','.join(MODELS))
    p.add_argument('--groups', type=str, default='1002,983,221,821,856,991',
                   help='published group ids to trace in the fate stage')
    p.add_argument('--min_size', type=int, default=3)
    p.add_argument('--match_top', type=int, default=400,
                   help='largest N groups per side in the match stage')
    p.add_argument('--n_null', type=int, default=5)
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--align_subpath', type=str, default='align/max_100')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--out_path', type=str, default='../refs/revision/tables')
    args = p.parse_args()

    rev = os.path.join(g['save_path'], args.rev_subpath)
    tags = [t.strip() for t in args.models.split(',') if t.strip()]
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    os.makedirs(args.out_path, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    kw = drift.load_keywords(os.path.join(g['save_path'], args.run_subpath))
    sets = kw_sets(kw)
    labs = load_groupings(rev, g['save_path'], args.run_subpath,
                          args.align_subpath, tags)
    print(f'{len(kw):,} units, groupings: {list(labs)}  '
          f'({time.time()-t0:.1f}s)', flush=True)

    if 'coherence' in stages:
        print('\n=== coherence: are the re-derived groups keyword-coherent? ===')
        recs = []
        for name, lab in labs.items():
            obs, null = grouping_coherence(sets, lab, rng, args.n_null,
                                           args.min_size)
            v = np.array([x for x in obs.values() if np.isfinite(x)])
            recs.append({'space': name, 'n_groups': len(v),
                         'median_coherence': float(np.median(v)),
                         'q25': float(np.quantile(v, .25)),
                         'q75': float(np.quantile(v, .75)),
                         'shuffled_null': null,
                         'ratio_to_null': float(np.median(v)) / null,
                         'frac_above_null': float((v > null).mean())})
            print(f'  {name:20s} median {recs[-1]["median_coherence"]:.4f} '
                  f'vs null {null:.4f}  = {recs[-1]["ratio_to_null"]:5.2f}x  '
                  f'({recs[-1]["frac_above_null"]:.0%} of groups above)',
                  flush=True)
        df = pd.DataFrame(recs)
        df.to_csv(os.path.join(args.out_path,
                               'R2.1_armB2_coherence_draft.csv'), index=False)
        print(df.round(4).to_string(index=False))

    if 'match' in stages:
        print('\n=== match: membership vs content correspondence ===')
        recs = []
        pairs = [('published', t) for t in ['published_realigned'] + tags]
        pairs += [('minilm', 'mpnet'), ('minilm', 'bge'), ('mpnet', 'bge')]
        for a, b in pairs:
            if a not in labs or b not in labs:
                continue
            m, null = match_groupings(labs[a], labs[b], kw, args.min_size,
                                      args.match_top)
            m.insert(0, 'pair', f'{a}->{b}')
            recs.append({
                'a': a, 'b': b, 'n_pairs': len(m),
                'median_containment': float(m.containment_a.median()),
                'median_kw_overlap': float(m.kw_overlap.median()),
                'kw_overlap_null': null,
                'kw_ratio_to_null': (float(m.kw_overlap.median()) / null
                                     if null > 0 else np.inf),
                # the diagnostic: pairs that barely share units but describe
                # the same content
                'frac_low_member_high_kw': float(
                    ((m.containment_a < 0.5)
                     & (m.kw_overlap > max(null * 2, 0.05))).mean()),
            })
            m.to_csv(os.path.join(args.out_path,
                                  f'R2.1_armB2_match_{a}_{b}_draft.csv'),
                     index=False)
            print(f'  {a:20s} -> {b:20s} containment {recs[-1]["median_containment"]:.3f} | '
                  f'kw overlap {recs[-1]["median_kw_overlap"]:.3f} '
                  f'(null {null:.3f}, {recs[-1]["kw_ratio_to_null"]:.2f}x)',
                  flush=True)
        df = pd.DataFrame(recs)
        df.to_csv(os.path.join(args.out_path,
                               'R2.1_armB2_match_summary_draft.csv'), index=False)
        print(df.round(4).to_string(index=False))

    if 'fate' in stages:
        print('\n=== fate: where do named published groups end up? ===')
        want = [int(x) for x in args.groups.split(',') if x.strip()]
        pub = labs['published']
        out = []
        for coi in want:
            idx = np.flatnonzero(pub == coi)
            lab_pub = drift.group_label(kw, idx, n=5)
            print(f'\n  published {coi} (T={len(idx)}): {lab_pub}')
            for name, lab in labs.items():
                if name == 'published':
                    continue
                dest = lab[idx]
                cnt = Counter(int(c) for c in dest)
                nz = {c: n for c, n in cnt.items() if c >= 0}
                big = sorted(nz.items(), key=lambda kv: -kv[1])[:3]
                frag = len(nz)
                desc = '; '.join(
                    f'{c}(n={n}/{ (lab==c).sum() }) "{drift.group_label(kw, np.flatnonzero(lab==c), n=4)}"'
                    for c, n in big)
                noise = cnt.get(-1, 0)
                print(f'    {name:20s} -> {frag} group(s), {noise} to noise | {desc}')
                out.append({'coi': coi, 'T': len(idx), 'label': lab_pub,
                            'space': name, 'n_destinations': frag,
                            'to_noise': noise,
                            'largest_share': (big[0][1] / len(idx)) if big else 0.0,
                            'destinations': desc})
        pd.DataFrame(out).to_csv(
            os.path.join(args.out_path, 'R2.1_armB2_fate_draft.csv'), index=False)

    print(f'\nDone in {(time.time()-t0)/60:.1f} min -> {args.out_path}')


if __name__ == '__main__':
    main()
