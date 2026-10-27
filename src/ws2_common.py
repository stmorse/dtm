"""
WS2 shared helpers (R2.4, R2.2, R1.3): alignment, partition summaries, drift.

Agreement and matching measures are NOT here -- they live in `matching.py`
(owned by WS3) and WS2 imports them. This module only holds what WS2's
scripts need on top of that and of `drift.py` (owned by WS1).
"""

import numpy as np
import pandas as pd

import drift

# the published alignment (3_align.py, align/max_100)
UMAP_KW = dict(n_neighbors=15, n_components=10, metric='euclidean',
               init='spectral', min_dist=0.1, spread=1.0)
HDBSCAN_KW = dict(min_cluster_size=3, min_samples=None,
                  cluster_selection_epsilon=0.0, max_cluster_size=100,
                  metric='euclidean')


def embed(C, seed, **umap_kw):
    """Seeded UMAP of a centroid matrix (seeding forces single-threaded)."""
    import umap
    return umap.UMAP(**dict(UMAP_KW, **umap_kw), random_state=seed).fit_transform(C)


def cluster(Cu, **hdbscan_kw):
    """HDBSCAN on an embedding -> group labels (-1 = noise)."""
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(**dict(HDBSCAN_KW, **hdbscan_kw)).fit(Cu).labels_


def align(C, seed, umap_kw=None, hdbscan_kw=None):
    """UMAP + HDBSCAN with the published settings unless overridden."""
    return cluster(embed(C, seed, **(umap_kw or {})), **(hdbscan_kw or {}))


def partition_summary(labels, k=50, min_size=5, max_cluster_size=100):
    """
    Aggregate description of a grouping of (month, cluster) units. These are
    the statistics WS3 found stable across seeds and encoders, so they are the
    ones to compare across alignment variants.
    """
    labels = np.asarray(labels)
    ids, n = np.unique(labels[labels >= 0], return_counts=True)
    tested = drift.tested_groups(labels, min_size=min_size)
    per = np.arange(len(labels)) // k
    span, n_months, multi = [], [], 0
    for g in tested:
        p = per[labels == g]
        span.append(p.max() - p.min() + 1)
        n_months.append(len(np.unique(p)))
        multi += len(np.unique(p)) < len(p)
    size_t = n[np.isin(ids, tested)]
    return {'n_units': len(labels), 'n_groups': len(ids),
            'noise_frac': float((labels == -1).mean()),
            'size_median': float(np.median(n)), 'size_p90': float(np.percentile(n, 90)),
            'size_max': int(n.max()),
            'n_at_cap': int((n >= max_cluster_size).sum()) if max_cluster_size else 0,
            'n_tested': len(tested),
            'frac_units_tested': float(size_t.sum() / len(labels)),
            'T_median': float(np.median(size_t)),
            'months_median': float(np.median(n_months)),
            'span_median': float(np.median(span)),
            'frac_tested_multi_unit_month': multi / max(len(tested), 1)}


def drift_table(C, labels, k=50, n_perm=100000, seed=0, min_size=5):
    """
    delta / Lambda / p / q per tested group via WS1's locked call. Each group
    gets its own RNG stream [seed, coi], so a group's p does not depend on
    which other groups a grouping happens to contain.
    """
    rows = []
    for coi in drift.tested_groups(labels, min_size=min_size):
        X, per = drift.group_trajectory(C, labels, coi, k=k, within_month='asis')
        T = X.shape[0]
        r = max(1, min(5, (T - 2) // 2))
        lam, p = drift.perm_test(drift.group_basis(X, r), drift.lam_batch,
                                 n_perm=n_perm, rng=[seed, int(coi)])
        rows.append({'coi': int(coi), 'T': T, 'r': r,
                     'first_period': int(per[0]), 'last_period': int(per[-1]),
                     'delta': drift.displacement(X), 'lam': float(lam),
                     'p_lam': float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df['q_lam'] = drift.bh_qvalues(df.p_lam.values)
    return df
