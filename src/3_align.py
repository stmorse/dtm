"""
Load cluster centers
Align
Save model
"""

# this prevents FutureWarning's coming from a sklearn dependency
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import configparser
import os
import pickle
import time

import numpy as np
import umap
from sklearn.cluster import HDBSCAN

def align_clusters(
    model_path: str,
    tfidf_path: str,
    align_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    align_dim: int,
    align_method: str,
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
    with open(os.path.join(align_path, f'cu.npz'), 'wb') as f:
        np.savez_compressed(f, Cu=Cu, allow_pickle=False)
    with open(os.path.join(align_path, f'cu2d.npz'), 'wb') as f:
        np.savez_compressed(f, Cu2d=Cu2d, allow_pickle=False)

    print(f'Fitting alignment model ({align_method}) ... ({time.time()-t0:.2f})')
    model, n_clusters, labels = None, None, None
    if align_method == 'HDBSCAN':    
        model = HDBSCAN(
            min_cluster_size=3,
            min_samples=None,       # None defaults to min_cluster_size
            cluster_selection_epsilon=0.0,
            max_cluster_size=20,
            metric='euclidean',
            store_centers='both',   # centroid and medoid
        )
        model.fit(Cu)

        labels = model.labels_
        n_clusters = np.amax(labels)
    else:
        print(f'Error: align method ({align_method}) not supported. Exiting ...')
        return

    print(f'> Complete. Num clusters: {n_clusters}')

    print(f'> Saving ...')
    with open(os.path.join(align_path, f'align_model_{align_method}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(align_path, f'align_model_{align_method}_labels.npz'), 'wb') as f:
        np.savez_compressed(f, labels=labels, allow_pickle=False)

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
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        results += f'Group: {k} ({idx})\n'
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, default=1, required=False)
    parser.add_argument('--end-month', type=int, default=12, required=False)
    parser.add_argument('--align-dim', type=int, default=10, required=False)
    parser.add_argument('--align-method', type=str, default="HDBSCAN", required=False)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)

    align_clusters(
        model_path=os.path.join(subpath, 'models'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
        align_path=os.path.join(subpath, 'align'),
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        align_dim=args.align_dim,
        align_method=args.align_method,
    )