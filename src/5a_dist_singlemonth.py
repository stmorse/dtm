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
from sklearn.cluster import MiniBatchKMeans, KMeans, HDBSCAN, kmeans_plusplus
import dask.array as da

KMEANS_SEED = 313

def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n_clusters', type=int, default=50)
    parser.add_argument('--n_resamples', type=int, default=10)
    parser.add_argument('--align_dim', type=int, default=10)
    parser.add_argument('--use_zarr', type=int, default=1)
    parser.add_argument('--model', type=str, default="mbkm")
    parser.add_argument('--make_clusters', type=int, default=1)
    parser.add_argument('--n_chunks_per_batch', type=int, default=1)

    args = parser.parse_args()

    # n_chunks_per_batch  - currently only for zarr + mbkm

    subpath = os.path.join(g['save_path'], args.subpath)

    # ensure directories exist
    for subdir in ['models', 'align']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir), exist_ok=True)

    # augment args with paths
    setattr(args, "embed_path", g["embed_path"])
    setattr(args, "model_path", os.path.join(subpath, 'models'))
    setattr(args, "align_path", os.path.join(subpath, 'align'))

    print(f'CPU count              : {os.cpu_count()}')
    print(f'Time period            : {args.year}, {args.month}')
    print(f'Saving results to path : {args.subpath}\n')

    # start time (used in _log)
    global t0
    t0 = time.time()

    if args.make_clusters==1:
        cluster(args)
    align(args)

def _log(msg):
    t = time.time() - t0
    print(f"{msg} ... ({t:.2f})")

def cluster(args):
    t0 = time.time()
    year, month = args.year, f'{args.month:02}'
    model_path = args.model_path

    # will hold all kmeans models (one for each permutation)
    models = []
    
    # --- USING ZARR ---

    if args.use_zarr == 1:
        
        # load data
        _log(f'Loading embeddings {year}-{month} (Using zarr)')
        ddata = da.from_zarr(os.path.join(args.embed_path, f'embeddings_{year}-{month}.zarr'))
        K = ddata.shape[0]
        M = len(ddata.chunks[0])
        _log(f'Total {K} embeddings in {M} chunks')
        
        if args.model == "mbkm":
            
            i = 0  # tracks total chunks
            j = 0  # tracks num chunks in this batch
            batch = []
            for chunk in ddata.to_delayed().ravel():
                
                arr = chunk.compute()
                _log(f'> Consolidating chunk {i+1}/{M} ({arr.shape[0]})')
                batch.append(arr)
                j += 1

                # permute and fit, if this is our final chunk for batch
                # note we already incremented j so we're checking ==
                if j == args.n_chunks_per_batch:
                    batch = np.vstack(batch)
                    _log(f'>> Fitting batch ({batch.shape[0]})')

                    # we have a batch ready, iterate over every perm
                    # and fit the corresponding model
                    for p in range(args.n_resamples):
                        _log(f'>>> Permutation {p+1}/{args.n_resamples}')

                        # first time, all data, other times, sampled with replacement
                        idx = np.arange(batch.shape[0])
                        if p > 0:
                            idx = np.random.permutation(batch.shape[0])

                        # compute initial cluster centroids first pass
                        # and initialize model
                        # NOTE: this does different init for every perm
                        if j - 1 == i:
                            # we are doing kmeans++ separately so that
                            # it's easy to change code to doing this once for 
                            # all perms for testing
                            C0, _ = kmeans_plusplus(
                                batch[idx,:], 
                                n_clusters=args.n_clusters,
                                random_state=KMEANS_SEED
                            )

                            # initialize model
                            model = MiniBatchKMeans(
                                n_clusters=args.n_clusters,
                                init=C0,
                                compute_labels=False,  # don't save labels
                                random_state=KMEANS_SEED
                            )

                            models.append(model)

                        # fit this permutation's model to its version of the batch
                        models[p].partial_fit(batch[idx,:])

                    # reset batch
                    j = 0
                    batch = []
                
                i += 1

        elif args.model == "km":

            pass

            # TODO: haven't updated to be the same as MBKM

            # Consolidate all embeddings, we're doing this in one batch
            # embeddings = []
            # i = 0
            # for chunk in ddata.to_delayed().ravel():
            #     arr = chunk.compute()
            #     L = arr.shape[0]
                
            #     print(f'> Consolidating chunk {i} ({L}) ... ({time.time()-t0:.2f})')
                
            #     idx = np.arange(L)
            #     if p > 0:
            #         idx = np.random.permutation(L)

            #     # manually compute initial cluster centroids first pass
            #     if p == 0 and i == 0:
            #         centers_init, _ = kmeans_plusplus(
            #             arr, 
            #             n_clusters=args.n_clusters,
            #             random_state=KMEANS_SEED
            #         )
            #         C0 = centers_init.copy()
                
            #     embeddings.append(arr[idx,:])
            #     i += 1

            # embeddings = np.vstack(embeddings)

            # print(f"> Clustering (KM) ... ({time.time()-t0:.2f})")
            # model = KMeans(
            #     n_clusters=args.n_clusters, 
            #     init=C0,
            #     random_state=KMEANS_SEED,
            #     algorithm="lloyd"
            # )
            # model.fit(embeddings)

        else:
            raise ValueError(f"Model not recognized ({args.model}).")

        # iterate through all models and save centroids
        _log("\nSaving centroids")
        for p in range(args.n_resamples):
            # save just centroids
            cc_name = f'model_cc_{year}-{month}_{p}.npz'
            with open(os.path.join(model_path, cc_name), 'wb') as f:
                np.savez_compressed(
                    f, 
                    cc=models[p].cluster_centers_.copy(), 
                    allow_pickle=False
                )

            _log(f'> Centroids saved for perm {p}')

    # --- USING NPZ ---
    # TODO: not implemented
    else:
        pass


def align(args):
    # --- ALIGN TOPIC CENTROIDS ---
    
    year, month = args.year, f'{args.month:02}'
    Ck = args.n_clusters
    align_dim = args.align_dim
    model_path = args.model_path
    align_path = args.align_path

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
    main()

