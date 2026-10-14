"""
Shared drift statistics for topic-group trajectories (WS1, PLOS revision R1.1).

Conventions
- C is the stacked centroid matrix, row = period * k + cluster.
- Group labels come from the alignment step (HDBSCAN); -1 = noise.
- A group's members are ordered by row index (i.e. by period) unless stated.

Other workstreams import from here; keep function signatures stable.
"""

import os

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def yrmo_range(years=range(2006, 2022 + 1), months=range(1, 12 + 1)):
    """(year, 'MM') pairs in period order, matching the published run."""
    return [(yr, f'{mo:02}') for yr in years for mo in months]


def load_centroids(run_path, yrmo=None, k=50):
    """Stack monthly centroids from <run_path>/models/model_cc_YYYY-MM.npz."""
    yrmo = yrmo_range() if yrmo is None else yrmo
    C = []
    for year, month in yrmo:
        with open(os.path.join(run_path, 'models', f'model_cc_{year}-{month}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
        if cc.shape[0] != k:
            raise ValueError(f'{year}-{month}: expected {k} centroids, got {cc.shape[0]}')
        C.append(cc)
    return np.vstack(C)


def load_group_labels(align_path):
    """Topic-group labels (one per centroid row) from an align/ directory."""
    with open(os.path.join(align_path, 'align_model_HDBSCAN_labels.npz'), 'rb') as f:
        return np.load(f)['labels']


def tested_groups(labels, min_size=5):
    """Group ids with label != -1 and at least min_size members, ascending."""
    ids, counts = np.unique(labels, return_counts=True)
    return [g for g, n in zip(ids, counts) if g != -1 and n >= min_size]


# ---------------------------------------------------------------------------
# descriptive statistics
# ---------------------------------------------------------------------------

def displacement(X):
    """delta_k: distance between first and last member (rows of X in time order)."""
    return float(np.linalg.norm(X[-1] - X[0]))


def diameter(X):
    """Largest pairwise distance among members."""
    d = X[:, None, :] - X[None, :, :]
    return float(np.sqrt(np.max(np.einsum('ijk,ijk->ij', d, d))))


# ---------------------------------------------------------------------------
# published procedure (viz.ipynb cell 75 / 7_rw.py), kept for the baseline
# ---------------------------------------------------------------------------

def published_pca(C, n_components=45, seed=0):
    """Global PCA fit on all centroids, as in the published run.

    The original call used sklearn defaults, which choose the (unseeded)
    randomized solver at this size; we seed it. Lambda is invariant to
    rotations within the subspace, so only the subspace estimate matters.
    """
    return PCA(n_components=n_components, random_state=seed).fit_transform(C)


def lambda_published(X):
    """Lambda = (T-1)/2 * mu' pinv(S) mu from consecutive steps of X (T x r).

    Returns (Lambda, pinv(S)); S is the ddof=1 covariance of the observed steps.
    """
    T = X.shape[0]
    steps = np.diff(X, axis=0)
    mu = steps.mean(axis=0)
    P = np.linalg.pinv(np.cov(steps, rowvar=False, ddof=1))
    return 0.5 * (T - 1) * (mu @ P @ mu), P


def perm_test_published(X, n_perm=10000, rng=None):
    """Published permutation test for one group (rows of X in time order).

    Replicates viz.ipynb cell 75: shuffle the order of the *centroids* (not the
    steps), recompute the mean step, and reuse the observed pinv(S).
    p = mean(null > obs) (strict, no +1 correction), as published.

    Vectorised via the telescoping identity: the mean step of a sequence is
    (last - first) / (T - 1), so only the permuted endpoints are needed.

    Returns dict with lam, p (published formula), p_plus1 = (1+b)/(B+1), null.
    """
    rng = np.random.default_rng(rng)
    T = X.shape[0]
    lam, P = lambda_published(X)

    # endpoints of B uniform random permutations: an ordered pair of distinct indices
    first = rng.integers(0, T, size=n_perm)
    last = rng.integers(0, T - 1, size=n_perm)
    last = last + (last >= first)
    mu = (X[last] - X[first]) / (T - 1)
    null = 0.5 * (T - 1) * np.einsum('bi,ij,bj->b', mu, P, mu)

    # permutations that keep (or reverse) the observed endpoints tie with lam exactly;
    # compare with a relative tolerance so rounding does not decide ties
    gt, ge = _cmp(null, lam)
    return {
        'lam': float(lam),
        'p': int(gt.sum()) / n_perm,
        'p_plus1': (1 + int(ge.sum())) / (n_perm + 1),
        'null': null,
    }


def perm_exact_published(X):
    """Exact version of perm_test_published by enumerating endpoint pairs.

    Under a uniform permutation the (first, last) pair is uniform over the
    T(T-1) ordered pairs of distinct members, and the statistic depends only on
    that pair, so the permutation distribution is enumerable.

    Returns dict with lam, p_gt = P(null > lam) (published convention) and
    p_ge = P(null >= lam) (conventional; includes the observed ordering).
    """
    T = X.shape[0]
    lam, P = lambda_published(X)
    i, j = np.where(~np.eye(T, dtype=bool))
    mu = (X[j] - X[i]) / (T - 1)
    null = 0.5 * (T - 1) * np.einsum('bi,ij,bj->b', mu, P, mu)
    gt, ge = _cmp(null, lam)
    return {'lam': float(lam), 'p_gt': gt.mean(), 'p_ge': ge.mean(),
            'n_pairs': T * (T - 1)}


def _degenerate(X, tol=1e-10):
    """True if the points are numerically identical (spread below tol)."""
    return float(np.abs(X - X.mean(axis=0)).max()) <= tol


def _cmp(null, obs, rtol=1e-9):
    tol = rtol * max(abs(obs), 1e-300)
    return null > obs + tol, null >= obs - tol


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

def group_rows(labels, coi, k=50, within_month='asis', rng=None):
    """Row indices of group coi, one entry per time point, in time order.

    Returns (rows, periods): rows is a list of index arrays (length 1 unless
    averaging), periods the month of each time point. within_month handles
    groups with several centroids in one month:
      'asis'    - keep all, ordered by row index (i.e. by k-means cluster id
                  within a month; arbitrary but deterministic)
      'random'  - keep all, ties within a month broken uniformly at random
      'average' - one time point per month, averaging its centroids (T = n months)
    """
    idx = np.where(labels == coi)[0]
    per = idx // k
    if within_month == 'asis':
        return [idx[i:i + 1] for i in range(len(idx))], per
    if within_month == 'random':
        rng = np.random.default_rng(rng)
        order = np.lexsort((rng.random(len(idx)), per))
        return [idx[i:i + 1] for i in order], per[order]
    if within_month == 'average':
        months = np.unique(per)
        return [idx[per == m] for m in months], months
    raise ValueError(within_month)


def trajectory(X_all, rows):
    """T x d trajectory from group_rows output (averaging multi-row time points)."""
    return np.stack([X_all[r].mean(axis=0) for r in rows])


def group_trajectory(X_all, labels, coi, k=50, within_month='asis', rng=None):
    """Convenience wrapper: (trajectory, periods) for group coi."""
    rows, per = group_rows(labels, coi, k=k, within_month=within_month, rng=rng)
    return trajectory(X_all, rows), per


def group_basis(X, r):
    """Project a group's centroids onto their own top-r principal axes.

    Depends only on the set of centroids, not their order, so it is fixed
    under the permutation null.
    """
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:r].T


# ---------------------------------------------------------------------------
# corrected statistics (batched: Xb is B x T x d, one ordering per row)
# ---------------------------------------------------------------------------

def lam_batch(Xb):
    """Lambda = (T-1)/2 mu' S^-1 mu, with S re-estimated for each ordering.

    Needs d <= T-2 for S (ddof=1 covariance of T-1 steps) to be invertible.
    """
    T = Xb.shape[1]
    steps = np.diff(Xb, axis=1)
    mu = steps.mean(axis=1)
    D = steps - mu[:, None, :]
    S = np.einsum('bti,btj->bij', D, D) / (T - 2)
    try:
        z = np.linalg.solve(S, mu[..., None])[..., 0]
    except np.linalg.LinAlgError:
        z = np.einsum('bij,bj->bi', np.linalg.pinv(S), mu)
    return 0.5 * (T - 1) * np.einsum('bi,bi->b', mu, z)


def delta2_batch(Xb):
    """Squared endpoint displacement ||x_last - x_first||^2."""
    d = Xb[:, -1] - Xb[:, 0]
    return np.einsum('bi,bi->b', d, d)


def perm_test(X, stat, n_perm=10000, rng=None, max_elems=2e7):
    """Monte Carlo permutation test of temporal exchangeability for one group.

    Shuffles the time order of the rows of X (T x d), recomputes stat on each
    ordering, and returns (obs, p) with p = (1 + #{null >= obs}) / (B + 1).
    """
    rng = np.random.default_rng(rng)
    T, d = X.shape
    if _degenerate(X):
        # every centroid identical (e.g. a null run that reproduces the same
        # solution exactly): no movement, so no evidence against the null
        return 0.0, 1.0
    obs = stat(X[None])[0]
    chunk = max(1, int(max_elems // (T * d)))
    b, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = rng.permuted(np.tile(np.arange(T), (m, 1)), axis=1)
        _, ge = _cmp(stat(X[perms]), obs)
        b += int(ge.sum())
        done += m
    return float(obs), (1 + b) / (n_perm + 1)


def perm_test_minp(Xr, rs, n_perm=10000, rng=None, max_elems=2e7):
    """Min-p combination of Lambda over several projection dims (exploratory).

    Xr is T x r_max in a nested basis (e.g. group_basis(X, max(rs))); Lambda_r
    uses its first r columns. Each ordering (observed + B shuffles, the same
    shuffles for every r) gets a per-r permutation p-value from its rank, and
    m = min_r p_r. The final p is the permutation p of m_obs against the m of
    all orderings (Westfall-Young min-p), so it is exact under exchangeability.

    Returns (p_final, per-r p-values of the observed ordering).
    """
    rng = np.random.default_rng(rng)
    T = Xr.shape[0]
    rs = [r for r in rs if r <= T - 2]
    chunk = max(1, int(max_elems // (T * max(rs))))
    L = [np.array([[lam_batch(Xr[None, :, :r])[0] for r in rs]])]
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = rng.permuted(np.tile(np.arange(T), (m, 1)), axis=1)
        Xb = Xr[perms]
        L.append(np.stack([lam_batch(Xb[:, :, :r]) for r in rs], axis=1))
        done += m
    L = np.vstack(L)                     # (B+1) x len(rs), row 0 = observed
    N = L.shape[0]
    # per-r p of each ordering: #{orderings with Lambda >= its Lambda} / N
    P = np.empty_like(L)
    for j in range(L.shape[1]):
        s = np.sort(L[:, j])
        P[:, j] = (N - np.searchsorted(s, L[:, j] * (1 - 1e-9), side='left')) / N
    mins = P.min(axis=1)
    p_final = np.sum(mins <= mins[0] * (1 + 1e-9)) / N
    return float(p_final), dict(zip(rs, P[0]))


