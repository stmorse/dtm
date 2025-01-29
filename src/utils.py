import numpy as np
from scipy.stats import f

def manova_pillai_trace(X, labels):
    """
    Perform one-way MANOVA using Pillai's trace.
    X     : (N x d) data matrix
    labels: (N,) array of group labels
    Returns (pillai, F_approx, p_value).
    """
    # Basic shapes and group info
    X = np.asarray(X)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    n_groups = len(groups)
    N, d = X.shape

    # Group means, overall mean
    overall_mean = X.mean(axis=0)
    group_means = []
    ns = []
    for g in groups:
        Xg = X[labels == g]
        group_means.append(Xg.mean(axis=0))
        ns.append(len(Xg))
    group_means = np.array(group_means)
    ns = np.array(ns)
    
    # Between-group scatter matrix B
    B = np.zeros((d, d))
    for i, g in enumerate(groups):
        mean_diff = group_means[i] - overall_mean
        B += ns[i] * np.outer(mean_diff, mean_diff)

    # Within-group scatter matrix W
    W = np.zeros((d, d))
    for i, g in enumerate(groups):
        Xg = X[labels == g]
        mg = group_means[i]
        diffs = Xg - mg
        W += diffs.T @ diffs
    
    # Solve generalized eigenproblem for M = inv(W)B (or do M = W^-1 * B)
    # We'll do an eig on W^-1 B. For stability, use e.g. eigh on W^-1 B if invertible
    W_inv = np.linalg.inv(W)
    M = W_inv @ B
    eigvals = np.linalg.eigvals(M)
    
    # Pillai's trace = sum_{i=1..r} lambda_i / (1 + lambda_i) for all real eigenvalues
    # (We assume d <= n_groups*N, so W is invertible in this toy example.)
    # But for one-way MANOVA, "s" is min(d, n_groups-1).
    # We'll keep only the top 's' real parts of eigenvalues
    # in case of minor numerical imaginary parts.
    s = min(d, n_groups - 1)
    # Sort descending by real part just to be safe
    real_eigs = np.sort(eigvals.real)[::-1][:s]
    pillai = np.sum(real_eigs / (1 + real_eigs))
    
    # Approximate F-test for Pillai’s trace:
    #   m = 0.5 * (|p - g + 1| - 1)
    #   Napprox = 0.5 * (N - g - 1)
    #   F = [ (2*Napprox + s + 1)/(2*m + s + 1 ) ] * [ pillai / (s - pillai) ]
    #   df1 = s*(2*m + s + 1),   df2 = s*(2*Napprox + s + 1)
    p = d
    g = n_groups
    m = 0.5 * (abs(p - g + 1) - 1)
    Napprox = 0.5 * (N - g - 1)
    num = (2 * Napprox + s + 1)
    den = (2 * m + s + 1)
    
    F_stat = (num / den) * (pillai / (s - pillai))
    df1 = s * (2*m + s + 1)
    df2 = s * (2*Napprox + s + 1)
    
    # p-value
    p_value = 1 - f.cdf(F_stat, df1, df2)
    
    return pillai, F_stat, p_value