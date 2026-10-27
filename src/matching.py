"""
Shared partition-matching and agreement measures (WS3, used by WS2).

No I/O, no config.ini, no paths: numpy, pandas, scipy, sklearn only, so this
imports cleanly from any script in src/.

Every docstring states the measure's failure mode, copied from R2.1's "Measures
used, with failure modes" in refs/revision/response.md so the module and the
response say the same thing. Two cautions that apply across the module, both
learned the hard way on R2.1:

  Noise is not a cluster. HDBSCAN marks 25-30% of units -1, and sklearn's ARI
  treats -1 as one cluster of ~2,800 units, so an all-units ARI measures mostly
  the noise boundary: on R2.1 arm B2 it read 0.11 where agreement on the units
  both groupings place was 0.52. Drop noise pairwise (`pair_scores` does) AND
  report `noise_summary` alongside, because dropping it flatters a variant that
  pushes hard-to-place units into noise.

  Homogeneity and completeness are NOT chance-adjusted. At ~1,000 groups over
  ~6,000 units a size-preserving shuffle already scores ~0.69, so only the
  excess over a null is interpretable. Pass `rng` to `pair_scores` to get it.

Owner: WS3. WS2 imports only. API changes belong in the Cross-workstream notes.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.stats import binomtest
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import homogeneity_completeness_v_measure as hcv


def matched_purity(a, b, k):
    """
    Fraction of points that land in correspondingly matched clusters under the
    best one-to-one matching of labels. Returns (purity, k x k contingency).

    Unlike ARI this is a plain readable proportion, and it is not deflated by
    cluster-size imbalance -- but it ignores how the disagreements are
    distributed, and it cannot go below 1/k, so it never looks as bad as chance.

    For k-means labels only: both labelings must be in [0, k). Noise labels
    would index M[-1] and wrap silently, so they raise instead. Use
    `pair_scores` for HDBSCAN groupings, which also handles differing group
    counts; this function assumes a square 1-1 matching over the same k.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    for name, v in (('a', a), ('b', b)):
        if v.size and (v.min() < 0 or v.max() >= k):
            raise ValueError(
                f'matched_purity: {name} has labels outside [0, {k}) '
                f'(min {v.min()}, max {v.max()}). Noise labels wrap silently '
                f'here -- use pair_scores for groupings with noise.')
    M = np.zeros((k, k), dtype=np.int64)
    for x, y in zip(a, b):
        M[x, y] += 1
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(a), M


def pair_scores(a, b, rng=None, n_null=3):
    """
    The agreement battery for two labelings of the same units. Noise (-1) is
    dropped on both sides, so pair this with `noise_summary`: dropping noise
    makes agreement look better when a variant pushes hard-to-place units into
    noise rather than mis-grouping them.

    Return keys are fixed (R2.1 tables depend on them): n, n_a, n_b,
    purity_1to1, overlap_many_to_one_median, ari, ami, homogeneity, completeness.

    Failure modes, per measure:
      purity_1to1   readable proportion, but forces one-to-one, so a clean
                    2-way split of one group is scored as half wrong.
      many_to_one   allows merges, so it only ever rises; it cannot see
                    fragmentation and will call a shattered grouping good.
      ari           harsh with many small clusters: splitting every group in
                    two nearly zeroes it while barely moving mutual
                    information. Chance-adjusted (0 = chance).
      ami           chance-adjusted and tolerant of refinement, so it stays
                    high under exactly the splitting ARI punishes. Read both.
      homogeneity /
      completeness  the asymmetric pair, which is what separates merges from
                    splits -- equal values mean symmetric boundary reshuffling,
                    not refinement. NOT chance-adjusted: pass `rng` for the
                    size-preserving null, which runs ~0.69 at ~1,000 groups.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    m = (a >= 0) & (b >= 0)
    A, B = a[m], b[m]
    ua, ia = np.unique(A, return_inverse=True)
    ub, ib = np.unique(B, return_inverse=True)
    M = coo_matrix((np.ones(len(ia)), (ia, ib)),
                   shape=(len(ua), len(ub))).toarray()
    r, c = linear_sum_assignment(-M)
    m2o = M.max(1) / M.sum(1)
    h, comp, _ = hcv(A, B)
    out = dict(n=int(M.sum()), n_a=len(ua), n_b=len(ub),
               purity_1to1=M[r, c].sum() / M.sum(),
               overlap_many_to_one_median=float(np.median(m2o)),
               ari=ari(A, B), ami=ami(A, B),
               homogeneity=h, completeness=comp)
    if rng is not None and n_null:
        nh, nc, na, nm = [], [], [], []
        for _ in range(n_null):
            Bs = rng.permutation(B)
            x, y, _ = hcv(A, Bs)
            nh.append(x); nc.append(y)
            na.append(ari(A, Bs)); nm.append(ami(A, Bs))
        out['homogeneity_null'] = float(np.mean(nh))
        out['completeness_null'] = float(np.mean(nc))
        out['ari_null'] = float(np.mean(na))
        out['ami_null'] = float(np.mean(nm))
    return out


def match_many_to_one(a, b, weights=None, min_overlap=None):
    """
    For each group of `a`, the group of `b` holding most of its members.

    Pass UNFILTERED labels: `overlap` is computed over members that are
    non-noise in both (which is how R2.1 defines it), while `frac_to_noise` is
    computed over all of the group's members, so noise in `b` has to be visible
    here. Passing pre-filtered arrays makes `frac_to_noise` 0 by construction.

    Columns: g_a, g_b, overlap, n_a, n_a_core, frac_to_noise.
      n_a       members of g_a in the input
      n_a_core  those also non-noise in b -- the denominator of `overlap`
      overlap   largest share of n_a_core falling in a single b group

    `weights` gives a per-member weight (e.g. cluster size, when matching at
    the (month, cluster) unit level rather than the comment level), so unit-level
    numbers can be compared with comment-level ones. Default None is unweighted.

    Failure mode: many-to-one is not symmetric and only ever rises -- several
    groups of `a` may match the same `b` group, so a high median overlap is
    consistent with `b` having merged everything. Always run it both ways, or
    read it next to `pair_scores`'s ari/ami. `min_overlap` filters rows and is a
    selection on the outcome: report how many rows it dropped.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None:
        w = np.ones(len(a))
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(a):
            raise ValueError('weights must be one per member')
    rows = []
    for g in np.unique(a[a >= 0]):
        sel = a == g
        core = sel & (b >= 0)
        n_a = float(w[sel].sum())
        n_core = float(w[core].sum())
        if n_core > 0:
            vals = np.unique(b[core])
            tot = np.array([w[core & (b == v)].sum() for v in vals])
            j = int(tot.argmax())
            g_b, ov = int(vals[j]), float(tot[j] / n_core)
        else:
            g_b, ov = -1, np.nan
        rows.append({'g_a': int(g), 'g_b': g_b, 'overlap': ov,
                     'n_a': n_a, 'n_a_core': n_core,
                     'frac_to_noise': float((n_a - n_core) / n_a) if n_a else np.nan})
    df = pd.DataFrame(rows, columns=['g_a', 'g_b', 'overlap', 'n_a',
                                     'n_a_core', 'frac_to_noise'])
    if min_overlap is not None:
        df = df[df.overlap >= min_overlap].reset_index(drop=True)
    return df


def noise_summary(a, b):
    """
    Noise fractions and their agreement. Report this with every `pair_scores`:
    agreement on the units both labelings place says nothing about how many
    units each one refused to place, and a variant that dumps hard units into
    noise scores better on everything else.

    Failure mode: `jaccard_noise` is a set-overlap measure and is confounded
    with set size, like any Jaccard -- compare it only between variants whose
    noise fractions are similar, which `frac_noise_a`/`frac_noise_b` show.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = a == -1, b == -1
    either = (na | nb).sum()
    return dict(frac_noise_a=float(na.mean()), frac_noise_b=float(nb.mean()),
                frac_noise_either=float((na | nb).mean()),
                frac_noise_both=float((na & nb).mean()),
                jaccard_noise=float((na & nb).sum() / either) if either else np.nan,
                agreement_noise_or_not=float((na == nb).mean()))


def containment(sa, sb, mode='min'):
    """
    Overlap of two sets normalised so set size cannot cap it. **Use this, not
    Jaccard**, whenever the two sets differ in size: with |A|=18 and |B|=115,
    Jaccard cannot exceed 0.157 even under perfect nesting, which is how R2.1
    first mis-read nested significant sets as conflicting ones.

      mode='min' (default)  |A n B| / min(|A|,|B|)  -- "are these nested at all",
                            symmetric. This is what R2.1 arm A reports.
      mode='a'              |A n B| / |A|           -- directional. Needed to say
                            WHICH way a chain nests; min() hides the asymmetry
                            (arm B1: published in minilm is 0.556 directionally
                            while minilm in mpnet is 0.939).

    Failure mode: containment ignores the other set's size entirely, so a set
    contained in a very large one scores 1.0 with no evidence of agreement.
    Pair it with `enrichment_test` or with the two set sizes.
    """
    sa, sb = set(sa), set(sb)
    if not sa or not sb:
        return np.nan
    inter = len(sa & sb)
    return inter / (min(len(sa), len(sb)) if mode == 'min' else len(sa))


def enrichment_test(hit, tot, p0):
    """
    Is a correspondence rate above chance? One-sided binomial test of `hit` of
    `tot` against baseline rate `p0` (e.g. the other space's overall
    significance rate). Returns dict(hit, tot, rate, chance_rate, enrichment, p).

    This is the measure to use instead of reporting a raw overlap count, which
    is meaningless without the base rate: 24 of 77 looks like disagreement until
    you note that chance is 15%.

    Failure mode: the binomial assumes the `tot` trials are independent. Matched
    groups overlap in members and are drawn from one alignment, so the p-value
    is anti-conservative; treat it as a sanity check on the direction, not as a
    calibrated test. `enrichment` (rate / chance) is the number to report.
    """
    bt = binomtest(hit, tot, p0, alternative='greater')
    return dict(hit=int(hit), tot=int(tot), rate=hit / tot, chance_rate=p0,
                enrichment=(hit / tot) / p0 if p0 else np.inf,
                p=float(bt.pvalue))
