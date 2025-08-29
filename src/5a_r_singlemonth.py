import argparse
import configparser
import os
import time

import numpy as np
import dask.array as da

from utils_km import _kmeans_parallel_init, _kmeanspp_init, _lloyd_faiss

KMEANS_SEED = 1337

def fit_kmeans_faiss(ddata, n_clusters, init_method='kmeans++', max_iter=20, tol=1e-4, spherical=True, random_state=KMEANS_SEED):
    if init_method == 'kmeans||':
        C0 = _kmeans_parallel_init(ddata, n_clusters, spherical=spherical, random_state=random_state)
    else:
        C0 = _kmeanspp_init(ddata, n_clusters, spherical=spherical, random_state=random_state)
    C = _lloyd_faiss(ddata, C0, max_iter=max_iter, tol=tol, spherical=spherical)
    return C, C0

def cluster(args):
    t0 = time.time()
    year, month = args.year, f'{args.month:02}'
    n_clusters = args.n_clusters
    model_path = os.path.join(args.sub_path, 'models')
    os.makedirs(model_path, exist_ok=True)

    print(f'Loading embeddings {year}-{month} (Using zarr: {args.use_zarr}) ... ({time.time()-t0:.2f})')

    if args.use_zarr == 1:
        ddata = da.from_zarr(os.path.join(args.embed_path, f'embeddings_{year}-{month}.zarr'))
        K = ddata.shape[0]
        print(f'Total {K} embeddings in {len(ddata.chunks[0])} chunks')

        rng_global = np.random.RandomState(KMEANS_SEED)
        print(f'\nInitializing with {getattr(args, "init_method", "kmeans++")} ... ({time.time()-t0:.2f})')
        C_star, C0 = fit_kmeans_faiss(
            ddata,
            n_clusters,
            init_method=getattr(args, 'init_method', 'kmeans++'),
            max_iter=getattr(args, 'init_max_iter', 10),
            tol=getattr(args, 'init_tol', 1e-3),
            spherical=getattr(args, 'spherical', True),
            random_state=KMEANS_SEED
        )
        del C_star  # we only need C0 as fixed init across resamples

        chunks0 = list(ddata.to_delayed().ravel())
        for p in range(args.n_resamples):
            print(f'\nFitting sample {p} ... ({time.time()-t0:.2f})')
            if p == 0:
                chunks = chunks0
                perm_within = False
            else:
                perm = rng_global.permutation(len(chunks0))
                chunks = [chunks0[i] for i in perm]
                perm_within = True
            C = _lloyd_faiss(
                ddata,
                C0,
                max_iter=getattr(args, 'max_iter', 50),
                tol=getattr(args, 'tol', 1e-4),
                spherical=getattr(args, 'spherical', True),
                chunks=chunks,
                permute_within=perm_within,
                rng=np.random.RandomState(KMEANS_SEED + p)
            )
            cc_name = f'model_cc_{year}-{month}_{p}.npz'
            with open(os.path.join(model_path, cc_name), 'wb') as f:
                np.savez_compressed(f, cc=C.copy(), allow_pickle=False)
            print(f'> Centroids saved to {cc_name} ({time.time()-t0:.2f})')
    else:
        pass


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_path', type=str, default=g['embed_path'])
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n_clusters', type=int, default=50)
    parser.add_argument('--n_resamples', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--align_dim', type=int, default=10)
    parser.add_argument('--use_zarr', type=int, default=1)
    parser.add_argument('--make_clusters', type=int, default=1)
    args = parser.parse_args()

    args.sub_path = os.path.join(g['save_path'], args.sub_path)

    print(f'CPU count              : {os.cpu_count()}')
    print(f'Time period            : {args.year}, {args.month}')
    print(f'Saving results to path : {args.sub_path}\n')

    if args.make_clusters==1:
        cluster(args)
    # align(args)
