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
from sklearn.cluster import MiniBatchKMeans, HDBSCAN

KMEANS_SEED = 314

def main(args):
    t0 = time.time()
    year, month = args.year, f'{args.month:02}'
    n_clusters = args.n_clusters
    chunk_size = args.chunk_size
    model_path = os.path.join(args.sub_path, 'models')
    label_path = os.path.join(args.sub_path, 'labels')
    align_path = os.path.join(args.sub_path, 'align')

    print(f'CPU count              : {os.cpu_count()}')
    print(f'Time period            : {year}, {month}')
    print(f'Saving results to path : {args.sub_path}\n')

    # ----
    # FIT / LABEL
    # ----

    print(f'Creating new model (n_clusters: {n_clusters}) ... ({time.time()-t0:.2f})')
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=KMEANS_SEED)

    print(f'Loading embeddings {year}-{month} ... ({time.time()-t0:.2f})')
    with open(os.path.join(args.embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
        embeddings = np.load(f)['embeddings']

    L = len(embeddings)
    M = L // chunk_size   # num chunks

    # fit the model to P versions of the embeddings
    for p in range(args.n_resamples):
        # first time, all data, other times, sampled with replacement
        idx = np.arange(L)
        if p > 0:
            idx = np.random.choice(L, size=L, replace=True)

        # --- FIT ---

        # partial_fit over all chunks
        print(f'Fitting model (size: {L}) (chunks: {M+1}) ...')
        for i in range(0, M):
            j = i * chunk_size
            chunk = embeddings[idx[j]:idx[j + chunk_size]]

            print(f'> Fitting chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
            model.partial_fit(chunk)
            del chunk
            gc.collect()

        # fit on "leftovers"
        fidx = M * chunk_size
        if fidx < L:
            leftovers = embeddings[idx[fidx:]]
            model.partial_fit(leftovers)
            del leftovers
            gc.collect()

        # save model
        model_name = f'model_{year}-{month}_{p}.pkl'
        with open(os.path.join(model_path, model_name), 'wb') as f:
            joblib.dump(model, f)

        # save just centroids
        cc_name = f'model_cc_{year}-{month}_{p}.npz'
        with open(os.path.join(model_path, cc_name), 'wb') as f:
            np.savez_compressed(f, cc=model.cluster_centers_.copy(), allow_pickle=False)

        print(f'> Model saved to {model_name} ({time.time()-t0:.2f})')

        # --- LABEL ---

        labels = []
        print(f'Labeling (size: {L}) (chunks: {M+1}) ...')
        for i in range(0, M):
            j = i * chunk_size
            chunk = embeddings[idx[j]:idx[j + chunk_size]]

            print(f'> Labeling chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
            chunk_labels = model.predict(chunk)
            labels.append(chunk_labels)
            del chunk
            gc.collect()

        # label "leftovers"
        fidx = M * chunk_size
        if fidx < L:
            leftovers = embeddings[idx[fidx:]]
            chunk_labels = model.predict(leftovers)
            labels.append(chunk_labels)
            del leftovers
            gc.collect()

        labels = np.concatenate(labels)

        # Save labels
        print(f'> Saving labels ... ({time.time()-t0:.2f})')
        label_name = f'labels_{year}-{month}_{p}.npz'
        with open(os.path.join(label_path, label_name), 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)

    # --- ALIGN ---
    # TODO: for now doing in a notebook

    Ck = args.n_clusters
    align_dim = 10

    print(f'\nLoading cluster centers ...')
    C = []  
    for p in range(10):
        with open(os.path.join(model_path, f'model_cc_{year}-{month}_{p}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
            if Ck != cc.shape[0]: print(Ck, cc.shape[0])
            C.append(cc)
    C = np.vstack(C)
    print(f'> Complete. (shape: {C.shape}) ...')
    print(f'> Num time windows: {C.shape[0] / Ck}')

    print(f'Dimension reduction ... ')
    u_embedder = umap.UMAP(
        n_neighbors=15,
        n_components=align_dim,
        metric='euclidean',
        init='spectral',
        min_dist=0.1,
        spread=1.0
    )
    Cu = u_embedder.fit_transform(C)
    print(f'> Shape: {Cu.shape} ...')

    Cu2d = umap.UMAP(n_components=2).fit_transform(C)

    print(f'> Saving Cu and Cu2d ... ')
    with open(os.path.join(align_path, f'cu_{year}-{month}.npz'), 'wb') as f:
        np.savez_compressed(f, Cu=Cu, allow_pickle=False)
    with open(os.path.join(align_path, f'cu2d_{year}-{month}.npz'), 'wb') as f:
        np.savez_compressed(f, Cu2d=Cu2d, allow_pickle=False)

    print(f'Fitting alignment model (HDBSCAN) ... ({time.time()-t0:.2f})')
    model = HDBSCAN(
        min_cluster_size=3,
        min_samples=None,       # None defaults to min_cluster_size
        cluster_selection_epsilon=0.0,
        max_cluster_size=100,
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
        print(f'> {label}: {dist}')

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
    args = parser.parse_args()

    args.sub_path = os.path.join(g['save_path'], args.sub_path)

    main(args)
