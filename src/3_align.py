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
# import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

def align_clusters(
    model_path: str,
    tfidf_path: str,
    align_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]
    yrmo = [(yr, mo) for yr in years for mo in months]

    print(f'CPU count                 : {os.cpu_count()}')
    print(f'Aligning range            : {years}, {months}')
    print(f'Saving model to path      : {align_path}')
    
    # open cluster centers
    print(f'\nLoading cluster centers ... ({time.time()-t0:.2f})')
    C = []
    Ck = -1  # TODO: doesn't need to be the same every month
    for year, month in yrmo:
        with open(os.path.join(model_path, f'model_cc_{year}-{month}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
            Ck = cc.shape[0]
            C.append(cc)
    C = np.vstack(C)
    print(f'> Complete. (shape: {C.shape}) ... ({time.time()-t0:.2f})')
    print(f'> Num clusters in each time window: {Ck}')
    print(f'> Num time windows: {C.shape[0] / Ck}  (should be whole number)')

    # TODO: doesn't need to be the same every month
    if len(yrmo) != C.shape[0] // Ck:
        print('Does not match model sizes! Exiting')
        return

    # threshold for cluster cutoff
    thresh = 0.25

    # create dendrogram
    # print(f'Making dendrogram ... ({time.time()-t0:.2f})')
    # Z = linkage(C, 'ward')
    # fig = plt.figure(figsize=(10, 7))
    # dendrogram(Z)
    # plt.axhline(y=thresh, color='r', linestyle='--')
    # fig.savefig('dendrogram.png', dpi=fig.dpi)  

    # cluster centroids
    print(f'Fitting alignment model ... ({time.time()-t0:.2f})')
    ahc = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.25,
        linkage='ward',
    )
    ahc.fit(C)
    print(f'> Model fit complete, (n_clusters: {ahc.n_clusters_})')
    print(f'> (Note: compare with clusters per time window, {Ck})')

    print(f'> Saving ...')
    with open(os.path.join(align_path, 'align_model.pkl'), 'wb') as f:
        joblib.dump(ahc, f)
    with open(os.path.join(align_path, 'align_model_labels.pkl'), 'wb') as f:
        np.savez_compressed(f, labels=ahc.labels_, allow_pickle=False)

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