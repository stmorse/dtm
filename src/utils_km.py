import os, time
import numpy as np
import dask.array as da
import faiss

KMEANS_SEED = 1337

def _l2_norm_inplace(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < eps] = 1.0
    x /= n

def _build_index(C):
    d = C.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(C.astype(np.float32, copy=False))
    return index

def _kmeanspp_init(ddata, k, spherical=True, random_state=KMEANS_SEED):
    rng = np.random.RandomState(random_state)
    K = ddata.shape[0]
    d = ddata.shape[1]
    C = np.empty((k, d), dtype=np.float32)
    i0 = int(rng.randint(0, K))
    C[0] = ddata[i0].compute().astype(np.float32, copy=False)
    if spherical: _l2_norm_inplace(C[[0]])

    for t in range(1, k):
        index = _build_index(C[:t] if not spherical else C[:t] / np.linalg.norm(C[:t], axis=1, keepdims=True))
        best_s = -1e9
        best_vec = None
        for chunk in ddata.to_delayed().ravel():
            X = chunk.compute().astype(np.float32, copy=False)
            if spherical: _l2_norm_inplace(X)
            D, _ = index.search(X, 1)
            w = D.ravel().astype(np.float64, copy=False)
            u = rng.random(w.shape[0])
            s = np.log(u + 1e-18) / np.maximum(w, 1e-18)
            j = int(np.argmax(s))
            if s[j] > best_s:
                best_s = float(s[j])
                best_vec = X[j].copy()
        C[t] = best_vec
    if spherical: _l2_norm_inplace(C)
    return C

def _kmeans_parallel_init(ddata, k, r_rounds=5, l_oversample=None, spherical=True, random_state=KMEANS_SEED):
    rng = np.random.RandomState(random_state)
    K = ddata.shape[0]
    d = ddata.shape[1]
    i0 = int(rng.randint(0, K))
    seeds = [ddata[i0].compute().astype(np.float32, copy=False)]
    if spherical: _l2_norm_inplace(seeds[0][None, :])
    if l_oversample is None: l_oversample = max(2*k, 64)

    for _ in range(r_rounds):
        C = np.vstack(seeds).astype(np.float32, copy=False)
        if spherical: _l2_norm_inplace(C)
        index = _build_index(C)
        phi = 0.0
        for chunk in ddata.to_delayed().ravel():
            X = chunk.compute().astype(np.float32, copy=False)
            if spherical: _l2_norm_inplace(X)
            D, _ = index.search(X, 1)
            phi += float(D.sum())

        cand = []
        for chunk in ddata.to_delayed().ravel():
            X = chunk.compute().astype(np.float32, copy=False)
            if spherical: _l2_norm_inplace(X)
            D, _ = index.search(X, 1)
            p = (l_oversample * D.ravel().astype(np.float64)) / max(phi, 1e-18)
            u = rng.random(p.shape[0])
            m = u < np.minimum(1.0, p)
            if m.any(): cand.append(X[m])
        if cand:
            seeds.append(np.vstack(cand).astype(np.float32, copy=False))

    Cc = np.vstack(seeds).astype(np.float32, copy=False)
    if spherical: _l2_norm_inplace(Cc)
    # shrink candidates to k via local k-means++ over candidates
    return _kmeanspp_on_array(Cc, k, spherical=spherical, random_state=random_state)

def _kmeanspp_on_array(X, k, spherical=True, random_state=KMEANS_SEED):
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    C = np.empty((k, d), dtype=np.float32)
    i0 = int(rng.randint(0, n))
    C[0] = X[i0]
    for t in range(1, k):
        index = _build_index(C[:t])
        D, _ = index.search(X, 1)
        w = D.ravel().astype(np.float64, copy=False)
        u = rng.random(n)
        s = np.log(u + 1e-18) / np.maximum(w, 1e-18)
        j = int(np.argmax(s))
        C[t] = X[j]
    if spherical: _l2_norm_inplace(C)
    return C

def _lloyd_faiss(ddata, C0, max_iter=20, tol=1e-4, spherical=True, chunks=None, permute_within=False, rng=None):
    C = C0.astype(np.float32, copy=True)
    if spherical: _l2_norm_inplace(C)
    k, d = C.shape
    for it in range(max_iter):
        index = _build_index(C)
        S = np.zeros((k, d), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        it_chunks = chunks if chunks is not None else ddata.to_delayed().ravel()
        for chunk in it_chunks:
            X = chunk.compute().astype(np.float32, copy=False)
            if permute_within and rng is not None:
                X = X[rng.permutation(X.shape[0])]
            if spherical: _l2_norm_inplace(X)
            D, I = index.search(X, 1)
            lab = I.ravel()
            counts += np.bincount(lab, minlength=k)
            for j in np.unique(lab):
                S[j] += X[lab == j].astype(np.float64, copy=False).sum(axis=0)
        newC = C.copy()
        m = counts > 0
        newC[m] = (S[m] / counts[m][:, None]).astype(np.float32, copy=False)
        if spherical: _l2_norm_inplace(newC)
        delta = np.linalg.norm((newC - C).astype(np.float64), axis=1).max()
        C = newC
        if delta <= tol: break
    return C