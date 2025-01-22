"""
Load cluster centers
Group (for now: AHC)
Save model
"""

import configparser
import os
import pickle
import time

import numpy as np
import umap
from sklearn.cluster import HDBSCAN
import joblib

def align_clusters(
    model_path: str,
    tfidf_path: str,
    align_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    align_dim: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]
    yrmo = [(yr, mo) for yr in years for mo in months]

    # TODO: doesn't need to be the same every month, hardcoded
    Ck = 50  

    print(f'CPU count                 : {os.cpu_count()}')
    print(f'Aligning range            : {years}, {months}')
    print(f'Saving model to path      : {align_path}')
    print(f'Cluster size:             : {Ck}')
    
    # open cluster centers
    print(f'\nLoading cluster centers ... ({time.time()-t0:.2f})')
    C = []  
    for year, month in yrmo:
        with open(os.path.join(model_path, f'model_cc_{year}-{month}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
            if Ck != cc.shape[0]: print(Ck, cc.shape[0])
            C.append(cc)
    C = np.vstack(C)
    print(f'> Complete. (shape: {C.shape}) ... ({time.time()-t0:.2f})')
    print(f'> Num time windows: {C.shape[0] / Ck}  (should be whole number)')

    # cluster centroids
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

    print(f'Fitting alignment model ... ({time.time()-t0:.2f})')
    hdbs = HDBSCAN(
        min_cluster_size=3,
        min_samples=None,       # None defaults to min_cluster_size
        cluster_selection_epsilon=0.0,
        max_cluster_size=20,
        metric='euclidean',
        store_centers='both',   # centroid and medoid
    )
    hdbs.fit(Cu)
    print(f'> Complete. Labels/counts: ... ')
    print(np.unique(hdbs.labels_, return_counts=True))

    print(f'> Saving ...')
    with open(os.path.join(align_path, 'align_model.pkl'), 'wb') as f:
        joblib.dump(hdbs, f)
    with open(os.path.join(align_path, 'align_model_labels.pkl'), 'wb') as f:
        np.savez_compressed(f, labels=hdbs.labels_, allow_pickle=False)

    # load cluster representations
    print(f'Loading tfidf ... ({time.time()-t0:.2f})')
    T = []
    for year, month in [(yr, mo) for yr in years for mo in months]:
        with open(os.path.join(tfidf_path, f'tfidf_{year}-{month}.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
            # TODO: hacky, maybe 'tfidf' shouldn't be a dict
            for k in range(len(tfidf['tfidf'].keys())):
                T.append(tfidf['tfidf'][k]['keywords'])
    
    # save aligned cluster representations
    print(f'Saving group representations ... ({time.time()-t0:.2f})')
    results = 'Cluster (Time): Keywords\n\n'
    for k in range(ahc.n_clusters_):
        idx = np.where(ahc.labels_ == k)[0]
        results += f'Group: {idx}\n'
        for x in idx:
            results += f'{x} ({x // Ck}): {T[x][:10]}\n'
        results += '\n'
    with open(os.path.join(align_path, f'results.log'), 'w') as f:
        f.write(results)

    print(f'Complete. ({time.time()-t0:.2f})')

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    # c = config['align']

    align_clusters(
        model_path=os.path.join(g['save_path'], g['run_subpath'], 'models'),
        tfidf_path=os.path.join(g['save_path'], g['run_subpath'], 'tfidf'),
        align_path=os.path.join(g['save_path'], g['run_subpath'], 'align'),
        start_year=int(g['start_year']),
        end_year=int(g['end_year']),
        start_month=int(g['start_month']),
        end_month=int(g['end_month']),
    )