import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import umap
import dask.array as da

from utils import manova_pillai_trace


def manova():
    print('loading group labels ...')

    with open('/sciclone/geograd/stmorse/reddit/mbkm_50/align/align_model_HDBSCAN_labels.pkl', 'rb') as f:
        labels = np.load(f)['labels']

    # specify group of interest
    goi = 6  # israel

    # number of clusters (and therefore number centroids in each time_period)
    Ck = 50

    # idx of group goi
    # example: 11 -> cluster 11 in time period 0
    # example: 56 -> cluster 6  in time period 1
    gidx = np.where(labels == goi)[0]

    print('gidx: ', gidx)

    print('total clusters in this group: ', len(gidx))

    # get cluster labels and time period
    cluster_labels = gidx % Ck
    time_periods = gidx // Ck

    print('cluster_labels ', cluster_labels)
    print('time_periods ', time_periods)

    start_year, end_year = 2006, 2008
    yrmo = [(yr, mo) for yr in range(start_year, end_year+1) for mo in range(1,13)]

    print('total time periods: ', len(yrmo))

    # loop on time periods
    # `all_samples` will be 100-line chunks of sample embeddings corresponding to cluster_labels
    all_samples = []
    for i, (yr, mo) in enumerate(yrmo):
        print('building ', yr, mo)

        # if no clusters from this group in this time period, skip
        if i not in time_periods:
            print('  ..skipping')
            continue

        with open(f'/sciclone/geograd/stmorse/reddit/mbkm_50/tfidf/tfidf_{yr}-{mo:02}.pkl', 'rb') as f:
            tfidf = pickle.load(f)

        # get indices in time_periods where this time period occurs
        # so we can get the associated label in cluster_label
        cix = np.where(time_periods == i)[0]

        # loop on cluster label
        all_sample_idx = []
        for ix in cix:
            cl = cluster_labels[ix]

            # get the top 100 closest comments to the centroid in this cluster
            sample_idx = tfidf['tfidf'][cl]['sample_indices']
            all_sample_idx.append(sample_idx)
        all_sample_idx = np.concatenate(all_sample_idx).astype(int)

        # now we need to extract all these embeddings
        print('  extract embeddings ... ')
        filepath, ex = None, None
        for ext in ['.zarr', '.npz']:
            potential_path = f'/sciclone/geograd/stmorse/reddit/embeddings/embeddings_{yr}-{mo:02}{ext}'
            if os.path.exists(potential_path):
                filepath = potential_path
                ex = ext
                break
        if filepath is None:
            raise FileNotFoundError(
                f'No file found for {yr}-{mo} with .zarr or .npz'
            )
            return
                
        embeddings = []
        if ex=='.npz':
            with open(filepath, 'rb') as f:
                embeddings = np.load(f)['embeddings']
        elif ex=='.zarr':
            ddata = da.from_zarr(filepath)
            embeddings = []
            for chunk in ddata.to_delayed().ravel():
                arr = chunk.compute()
                embeddings.append(arr)
            embeddings = np.vstack(embeddings)
        else:
            print('should not get here')
            return
        print('  embeddings ', embeddings.shape)

        all_samples.append(embeddings[all_sample_idx,:])

    all_samples = np.vstack(all_samples)
    print('all_samples ', all_samples.shape)

    # project to d=10
    u_embedder = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        metric='euclidean',
        init='spectral',
        min_dist=0.1,
        spread=1.0
    )
    asu = u_embedder.fit_transform(all_samples)
    print('dim reduce shape ', asu.shape)

    # now we expand cluster labels (use gidx) to be same size as all_samples
    all_samples_labels = np.repeat(gidx, 100)
    print('all_samples_labels ', all_samples_labels.shape)

    # compute manova
    print('computing manova ...')
    pillai, F_stat, p_val = manova_pillai_trace(asu, all_samples_labels)

    print(f"Pillai's Trace: {pillai:.4f}")
    print(f"Approx F-Statistic: {F_stat:.4f}")
    print(f"p-value: {p_val:.6f}")


if __name__=="__main__":
    os.environ['PYTHONUNBUFFERED'] = '1'
    manova()