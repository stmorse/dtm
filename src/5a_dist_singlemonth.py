"""
Load single month
Cluster / bootstrap cluster
Align
Save model
"""

# this prevents FutureWarning's coming from a sklearn dependency
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import configparser
import gc
import os
import pickle
import time

import joblib
import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans, KMeans, HDBSCAN
import dask.array as da

KMEANS_SEED = 313

def cluster(args):
    t0 = time.time()
    year, month = args.year, f'{args.month:02}'
    n_clusters = args.n_clusters
    chunk_size = args.chunk_size
    model_path = os.path.join(args.sub_path, 'models')
    # label_path = os.path.join(args.sub_path, 'labels')

    # ----
    # FIT / LABEL
    # ----

    print(
        f'Loading embeddings {year}-{month} (Using zarr: {args.use_zarr}) '
        f'... ({time.time()-t0:.2f})'
    )
    
    # --- USING ZARR ---

    if args.use_zarr:
        # load data
        ddata = da.from_zarr(os.path.join(args.embed_path, f'embeddings_{year}-{month}.zarr'))
        K = ddata.shape[0]
        print(f'Total {K} embeddings in {len(ddata.chunks[0])} chunks')

        for p in range(args.n_resamples):
            print(f'\nFitting sample {p} ... ({time.time()-t0:.2f})')
            
            # --- FIT ---

            # reinitialize a model
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=KMEANS_SEED)

            i = 0
            for chunk in ddata.to_delayed().ravel():
                arr = chunk.compute()
                L = arr.shape[0]
                print(f'> Fitting chunk {i} ({L}) ... ({time.time()-t0:.2f})')

                # first time, all data, other times, sampled with replacement
                idx = np.arange(L)
                if p > 0:
                    idx = np.random.permutation(L)

                model.partial_fit(arr[idx,:])
                i += 1

            # save just centroids
            cc_name = f'model_cc_{year}-{month}_{p}.npz'
            with open(os.path.join(model_path, cc_name), 'wb') as f:
                np.savez_compressed(f, cc=model.cluster_centers_.copy(), allow_pickle=False)

            print(f'> Centroids saved to {cc_name} ({time.time()-t0:.2f})')

    # --- USING NPZ ---

    else:
        pass


def align(args):
    # --- ALIGN TOPIC CENTROIDS ---
    
    year, month = args.year, f'{args.month:02}'
    Ck = args.n_clusters
    align_dim = args.align_dim
    model_path = os.path.join(args.sub_path, 'models')
    align_path = os.path.join(args.sub_path, 'align')

    t0 = time.time()

    print(f'\nLoading cluster centers ...')
    C = []  
    for p in range(args.n_resamples):
        with open(os.path.join(model_path, f'model_cc_{year}-{month}_{p}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
            if Ck != cc.shape[0]: print(Ck, cc.shape[0])
            C.append(cc)
    C = np.vstack(C)
    print(f'> Complete. (shape: {C.shape}) ...')
    print(f'> Num time windows: {C.shape[0] / Ck}')

    print(f'Dimension reduction ... ')
    u_embedder = umap.UMAP(
        n_neighbors=15,  # default 15
        n_components=align_dim,
        metric='euclidean',
        init='spectral',
        min_dist=0.1,  # default is 0.1
        spread=1.0   # default is 1.0
    )
    Cu = u_embedder.fit_transform(C)
    print(f'> Shape: {Cu.shape} ...')

    Cu2d = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        spread=1.0,
    ).fit_transform(C)

    print(f'> Saving Cu and Cu2d ... ')
    with open(os.path.join(align_path, f'cu_{year}-{month}.npz'), 'wb') as f:
        np.savez_compressed(f, Cu=Cu, allow_pickle=False)
    with open(os.path.join(align_path, f'cu2d_{year}-{month}.npz'), 'wb') as f:
        np.savez_compressed(f, Cu2d=Cu2d, allow_pickle=False)

    print(f'Fitting alignment model (HDBSCAN) ... ({time.time()-t0:.2f})')
    model = HDBSCAN(
        min_cluster_size=5,
        min_samples=None,       # None defaults to min_cluster_size
        cluster_selection_epsilon=0.0,
        max_cluster_size=300,
        metric='euclidean',
        store_centers='both',   # centroid and medoid
    )
    model.fit(Cu)

    # this is all labels for Cu from .fit
    labels = model.labels_
    n_align_clusters = np.amax(labels)  # note HDBSCAN includes a -1 cluster
    print(f'> Complete. Num clusters: {n_align_clusters}')
    print(f'> Size of outliers: {len(np.where(labels == -1)[0])}')

    print(f'> Saving ...')
    with open(os.path.join(align_path, f'align_model_{year}-{month}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(align_path, f'align_model_{year}-{month}_labels.npz'), 'wb') as f:
        np.savez_compressed(f, labels=labels, allow_pickle=False)

    print(f'Computing spread in each group:')
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        cut = Cu[idx,:]
        mid = np.mean(cut, axis=0)
        dist = np.mean(np.linalg.norm(cut - mid, axis=1))
        print(f'> {label} ({len(idx)}): {dist}')

    print(f'Complete ... ({time.time()-t0:.2f})')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=g['data_path']) # raw data
    parser.add_argument('--embed_path', type=str, default=g['embed_path'])
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n_clusters', type=int, default=50)
    parser.add_argument('--n_resamples', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--align_dim', type=int, default=10)
    parser.add_argument('--use_zarr', type=int, default=0)
    parser.add_argument('--make_clusters', type=int, default=1)
    args = parser.parse_args()

    args.sub_path = os.path.join(g['save_path'], args.sub_path)

    print(f'CPU count              : {os.cpu_count()}')
    print(f'Time period            : {args.year}, {args.month}')
    print(f'Saving results to path : {args.sub_path}\n')

    if args.make_clusters==1:
        cluster(args)
    align(args)

