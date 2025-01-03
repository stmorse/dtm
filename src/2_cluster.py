"""
Loads embeddings into memory
Trains cluster model on every month
Labels that month
Saves to file
"""

import argparse
import configparser
import gc
import time
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib

def train_cluster_model(
    embed_path: str,
    label_path: str,
    model_path: str,
    n_clusters: int,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    chunk_size: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {label_path}\n')
    
    
    print(f'Creating new model ... ({time.time()-t0:.2f})')
    # we're using partial_fit on chunks so I think batch_size is irrelevant
    model = MiniBatchKMeans(n_clusters=n_clusters)
    
    # model_path = os.path.join(label_path, f'{model_name}.pkl')
    # print(f'Loading model from {model_path} ... ({time.time()-t0:.2f})')
    # model = joblib.load(model_path)

    print(f'Loaded model (n_clusters: {model.n_clusters}) ... ({time.time()-t0:.2f})')

    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Loading embeddings {year}-{month} ... ({time.time()-t0:.2f})')
        with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
            embeddings = np.load(f)['embeddings']
            
            # ----
            # FIT
            # ----

            # partial_fit over all chunks
            L = len(embeddings)
            M = L // chunk_size   # num chunks
            print(f'> Fitting (size: {L}) (chunks: {M+1}) ...')
            for i in range(0, M):
                j = i * chunk_size
                chunk = embeddings[j:j + chunk_size]

                print(f'> Fitting chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
                model.partial_fit(chunk)
                del chunk
                gc.collect()

            # fit on "leftovers"
            fidx = M * chunk_size
            if fidx < L:
                leftovers = embeddings[fidx:]
                model.partial_fit(leftovers)
                del leftovers
                gc.collect()

            # save model (folder name captures type/n_clusters)
            model_name = f'model_{year}-{month}.pkl'
            with open(os.path.join(model_path, model_name), 'wb') as f:
                joblib.dump(model, f)

            # save just cluster centroids (for joblib incompatibility)
            cc_name = f'model_cc_{year}-{month}.npz'
            with open(os.path.join(model_path, cc_name), 'wb') as f:
                np.savez_compressed(f, cc=model.cluster_centers_.copy(), allow_pickle=False)

            print(f'Cluster model saved to {model_path} ({time.time()-t0:.2f})')

            # ----
            # LABEL
            # ----

            labels = []
            print(f'> Labeling (size: {L}) (chunks: {M+1}) ...')
            for i in range(0, M):
                j = i * chunk_size
                chunk = embeddings[j:j + chunk_size]

                print(f'> Labeling chunk {i} ({len(chunk)}) ... ({time.time()-t0:.2f})')
                chunk_labels = model.predict(chunk)
                labels.append(chunk_labels)
                del chunk
                gc.collect()

            # label "leftovers"
            fidx = M * chunk_size
            if fidx < L:
                leftovers = embeddings[fidx:]
                chunk_labels = model.predict(leftovers)
                labels.append(chunk_labels)
                del leftovers
                gc.collect()

            labels = np.concatenate(labels)
            print(f'> Labeling complete ({labels.shape}) ... ({time.time()-t0:.2f})')

            # Save labels
            print(f'> Saving labels ... ({time.time()-t0:.2f})')
            label_name = f'labels_{year}-{month}.npz'
            with open(os.path.join(label_path, label_name), 'wb') as f:
                np.savez_compressed(f, labels=labels, allow_pickle=False)

            del embeddings
            gc.collect()

    print(f'Complete. ({time.time()-t0:.2f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    c = config['cluster']

    train_cluster_model(
        embed_path=os.path.join(config['general']['save_path'], c['embed_subpath']),
        label_path=os.path.join(config['general']['save_path'], c['label_subpath']),
        model_path=os.path.join(config['general']['save_path'], c['model_subpath']),
        n_clusters=int(c['n_clusters']),
        start_year=int(c['start_year']),
        end_year=int(c['end_year']),
        start_month=int(c['start_month']),
        end_month=int(c['end_month']),
        chunk_size=int(c['chunk_size']),
    )