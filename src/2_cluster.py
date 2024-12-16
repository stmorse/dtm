"""
Loads embeddings into memory
Trains cluster model 
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
    n_clusters: int,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    batch_size: int,
    model_name: str,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {label_path}\n')
    
    if model_name is "none":
        print(f'Creating new model ... ({time.time()-t0:.2f})')
        # we're using partial_fit on chunks so I think this batch_size is irrelevant
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    else:
        model_path = os.path.join(label_path, f'{model_name}.pkl')
        print(f'Loading model from {model_path} ... ({time.time()-t0:.2f})')
        model = joblib.load(model_path)
    
    print(f'Loaded model (n_clusters: {model.n_clusters}) ... ({time.time()-t0:.2f})')

    print(f'Loading embeddings ... ({time.time()-t0:.2f})')
    for year, month in [(yr, mo) for yr in years for mo in months]:
        with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
            embeddings = np.load(f)
            print(f'Loaded embeddings_{year}-{month}.npz ({embeddings.shape}) ... ({time.time()-t0:.2f})')
            model.partial_fit(embeddings)
            print(f'Partial fit done ... ({time.time()-t0:.2f})')

    # load embeddings
    embeddings = np.load(label_path)
    print(f'Embeddings shape: {embeddings.shape}')

    print(f'Training cluster model ... ({time.time()-t0:.2f})')

    # train cluster model
    kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=batch_size, verbose=1)
    kmeans.fit(embeddings)

    # save model
    model_path = os.path.join(os.path.dirname(label_path), 'cluster_model.pkl')
    joblib.dump(kmeans, model_path)

    print(f'Cluster model saved to {model_path} ({time.time()-t0:.2f})')

    del embeddings
    gc.collect()

    pass


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    c = config['cluster']

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clusters', type=int, default=c['n_clusters'])
    parser.add_argument('--start-year', type=int, default=c['start_year'])
    parser.add_argument('--end-year', type=int, default=c['end_year'])
    parser.add_argument('--start-month', type=int, default=c['start_month'])
    parser.add_argument('--end-month', type=int, default=c['end_month'])
    parser.add_argument('--batch-size', type=int, default=c['batch_size'])
    parser.add_argument('--embed-path', type=str, default=config['general']['data_path'])
    parser.add_argument('--label-path', type=str, default=c['meta_subpath'])
    parser.add_argument('--load-model', type=bool, default=c['load_model'])
    
    args = parser.parse_args()

    train_cluster_model(
        embed_path=args.embed_path,
        label_path=os.path.join(config['general']['save_path'], args.label_path),
        n_clusters=args.n_clusters,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        batch_size=args.batch_size,
        load_model=args.load_model,
    )