import numpy as np
import faiss

KMEANS_SEED = 1337

# ------------------------
# PRIVATE METHODS
# ------------------------

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
    """kmeans++ initialization"""
    
    # init values
    rng = np.random.RandomState(random_state)
    K = ddata.shape[0]  # num samples
    d = ddata.shape[1]  # dimension
    tol = 1e-18

    # create an empty array with row = cluster centroid
    C = np.empty((k, d), dtype=np.float32)

    # pick a random row and make it the first cluster centroid 
    i0 = int(rng.randint(0, K))
    C[0] = ddata[i0].compute().astype(np.float32, copy=False)
    if spherical: 
        _l2_norm_inplace(C[[0]])

    # iterate over n_clusters (k)
    for t in range(1, k):
        
        # build FAISS index over centroids thus far
        index = _build_index(
            C[:t] if not spherical 
            else C[:t] / np.linalg.norm(C[:t], axis=1, keepdims=True)
        )

        # iterate over chunks and keep track of best
        best_s = -1e9
        best_vec = None
        for chunk in ddata.to_delayed().ravel():
            # extract chunk to float32
            X = chunk.compute().astype(np.float32, copy=False)
            
            if spherical: 
                _l2_norm_inplace(X)
            
            # search nearest neighbor (1) in index (over C[:t])
            # for each row of X, keep the distances (Dij = dist of X_i to C_j)
            D, _ = index.search(X, 1)

            # sample new center with prob \propto D(x)^2
            w = D.ravel().astype(np.float64, copy=False)
            u = rng.random(w.shape[0])
            s = np.log(u + tol) / np.maximum(w, tol)
            j = int(np.argmax(s))
            
            # if this is highest prob seen, assign it as new centroid
            if s[j] > best_s:
                best_s = float(s[j])
                best_vec = X[j].copy()
        
        C[t] = best_vec
    
    if spherical: 
        _l2_norm_inplace(C)
    
    return C

def _kmeans_parallel_init(
        ddata, k, r_rounds=5, l_oversample=None, spherical=True, 
        random_state=KMEANS_SEED
):
    """kmeans|| implementation"""

    # init values
    rng = np.random.RandomState(random_state)
    K = ddata.shape[0]
    d = ddata.shape[1]

    # TODO: finish commenting

    i0 = int(rng.randint(0, K))
    seeds = [ddata[i0].compute().astype(np.float32, copy=False)]
    if spherical: _l2_norm_inplace(seeds[0][None, :])
    if l_oversample is None: l_oversample = max(2*k, 64)

    for rd in range(r_rounds):
        print(f"Round {rd} ...")

        C = np.vstack(seeds).astype(np.float32, copy=False)
        if spherical: 
            _l2_norm_inplace(C)
        
        index = _build_index(C)
        phi = 0.0
        for chunk in ddata.to_delayed().ravel():
            X = chunk.compute().astype(np.float32, copy=False)
            print(f".. processing chunk ({X.shape})")
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

def _lloyd_faiss(
        ddata,                  # ignored if chunks is not None
        C0,                     # initial centroids
        max_iter=20,            # max iter of lloyd's algo
        tol=1e-4,               # tolerance for convergence
        spherical=True,         # apply l2 norming
        chunks=None,            # allows sending explicit chunks instead of ddata
        permute_within=False,   # permute chunks (TODO: not really needed...)
        rng=None
):
    """Lloyd's algorithm implementation, using FAISS for indexing"""

    # convert initial centroids to float32 and norm
    C = C0.astype(np.float32, copy=True)
    if spherical: 
        _l2_norm_inplace(C)
    k, d = C.shape

    # iterate to max (or convergence to tol, whichever comes first)
    print("K-means iteration: ", end="")
    for it in range(max_iter):
        if it % 5 == 0:
            print(f" {it} ", end="")
        # build index for current set of centroids
        index = _build_index(C)

        # 
        S = np.zeros((k, d), dtype=np.float64)

        # running count of each cluster
        counts = np.zeros(k, dtype=np.int64)    
        
        # iterate over chunks
        it_chunks = chunks if chunks is not None else ddata.to_delayed().ravel()
        for chunk in it_chunks:
            # extract chunk to array
            X = chunk.compute().astype(np.float32, copy=False)
            
            # TODO: only used for the bootstrap example ....
            if permute_within and rng is not None:
                X = X[rng.permutation(X.shape[0])]

            if spherical: 
                _l2_norm_inplace(X)
            
            # find nearest centroids to each row of X
            D, I = index.search(X, 1)
            lab = I.ravel()

            # update S and cluster counts
            counts += np.bincount(lab, minlength=k)
            for j in np.unique(lab):
                S[j] += X[lab == j].astype(np.float64, copy=False).sum(axis=0)
        
        newC = C.copy()
        m = counts > 0
        newC[m] = (S[m] / counts[m][:, None]).astype(np.float32, copy=False)
        
        if spherical: 
            _l2_norm_inplace(newC)
        
        delta = np.linalg.norm((newC - C).astype(np.float64), axis=1).max()
        C = newC
        
        if delta <= tol: 
            break

    print(f"Complete. ({it} iterations)")

    # return cluster centroids and the most recent index
    return C, index


# ------------------------
# PUBLIC METHODS
# ------------------------

def init_kmeans(
        ddata, n_clusters, 
        method='kmeans++', spherical=True, random_state=KMEANS_SEED, 
        r_rounds=5, l_oversample=None
):
    if method == 'kmeans||':
        return _kmeans_parallel_init(
            ddata, n_clusters, r_rounds=r_rounds, l_oversample=l_oversample, 
            spherical=spherical, random_state=random_state
        )
    else:
        return _kmeanspp_init(
            ddata, n_clusters, spherical=spherical, random_state=random_state
        )

def fit_kmeans_faiss(
        ddata, n_clusters, 
        C0=None, init_method='kmeans++', 
        max_iter=20, tol=1e-4, spherical=True, random_state=KMEANS_SEED
):
    # find initial centers if not specified
    if C0 is None:
        C0 = init_kmeans(
            ddata, n_clusters, method=init_method, 
            spherical=spherical, random_state=random_state
        )
        
    # run FAISS-enabled lloyd's algorithm
    C, index = _lloyd_faiss(
        ddata, C0, max_iter=max_iter, tol=tol, spherical=spherical)
    
    return C, C0, index

def predict_kmeans_faiss(X, C, index=None, spherical=True):
    """
    X (np array), C (centroids), index (FAISS index)
    """

    # copy centroids and normalize if needed
    C_ = C.astype(np.float32, copy=True)
    if spherical:
        _l2_norm_inplace(C_)

    # build FAISS index
    if index is None:
        index = _build_index(C_)

    # process data
    X = X.astype(np.float32, copy=False)
    if spherical:
        _l2_norm_inplace(X)

    # search nearest centroid
    _, I = index.search(X, 1)
    return I.ravel()

