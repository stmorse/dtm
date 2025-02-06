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


def hotelling_t2_test_single(X, mu0=None):
    """
    Conduct a one-sample Hotelling T^2 test for the hypothesis H0: E[X] = mu0.
    
    Parameters
    ----------
    X : (n, d) array-like
        The sample data. Each row is an observation in R^d.
    mu0 : (d,) array-like or None
        The hypothesized mean vector. If None, tests against a zero vector.
        
    Returns
    -------
    T2_stat : float
        The Hotelling T^2 statistic.
    F_stat : float
        The equivalent F-statistic.
    p_value : float
        The p-value from the F-distribution.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    
    if mu0 is None:
        mu0 = np.zeros(d)
    else:
        mu0 = np.asarray(mu0, dtype=float)
        if mu0.shape != (d,):
            raise ValueError("mu0 must be of shape (d,).")
    
    # Sample mean
    X_bar = X.mean(axis=0)

    # Sample covariance (unbiased) and inverse
    S = np.cov(X, rowvar=False)  # shape (d, d)
    # Might need a pseudo-inverse if S is singular in high dimensions
    S_inv = np.linalg.inv(S)
    
    # Hotelling T^2
    diff = X_bar - mu0
    T2_stat = n * diff @ S_inv @ diff
    
    # Convert to F-statistic
    # F ~ F(d, n-d) under H0
    F_stat = ((n - d) / (d * (n - 1))) * T2_stat
    
    # p-value
    p_value = 1 - f.cdf(F_stat, d, n - d)
    
    return T2_stat, F_stat, p_value



def hotelling_t2_test(X, Y):
    """
    Perform a two-sample Hotelling's T-squared test.
    
    Parameters:
        X: np.ndarray of shape (n1, p) - First sample group (n1 observations, p variables)
        Y: np.ndarray of shape (n2, p) - Second sample group (n2 observations, p variables)
    
    Returns:
        T2: Hotelling's T-squared statistic
        F_value: Corresponding F-statistic
        p_value: p-value for the test
    """
    X, Y = np.asarray(X), np.asarray(Y)
    n1, p = X.shape
    n2, _ = Y.shape
    
    # Compute sample means
    x_bar = np.mean(X, axis=0)
    y_bar = np.mean(Y, axis=0)
    
    # Compute sample covariance matrices
    S1 = np.cov(X, rowvar=False, ddof=1)
    S2 = np.cov(Y, rowvar=False, ddof=1)
    
    # Compute pooled covariance matrix
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    
    # Compute Hotelling's T-squared statistic
    mean_diff = x_bar - y_bar
    Sp_inv = np.linalg.inv(Sp)  # Inverse of pooled covariance matrix
    T2 = (n1 * n2) / (n1 + n2) * mean_diff.T @ Sp_inv @ mean_diff
    
    # Convert to F-statistic
    F_value = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * T2
    df1, df2 = p, (n1 + n2 - p - 1)
    
    # Compute p-value
    p_value = 1 - f.cdf(F_value, df1, df2)
    
    return T2, F_value, p_value