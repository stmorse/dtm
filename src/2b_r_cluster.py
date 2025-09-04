import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import logging
import os
import pickle
import time

import bz2
import lzma
import zstandard as zstd

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import dask.array as da
import dask_ml.cluster
from dask.distributed import Client, LocalCluster, wait, progress
from dask.diagnostics import ProgressBar

from utils import read_sentences
# from utils_km import init_kmeans, fit_kmeans_faiss, predict_kmeans_faiss


def train_cluster_model(
    args,
    data_path: str,
    embed_path: str,
    label_path: str,
    model_path: str,
    tfidf_path: str,   
):
    t0 = time.time()
    years = [str(y) for y in range(args.start_year, args.end_year+1)]
    months = [f'{m:02}' for m in range(args.start_month, args.end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {label_path}\n')

    # print(f"Initializing logging ...")
    # logging.getLogger("distributed").setLevel(logging.INFO)

    # print(f"Initializing scheduler ...")
    # cluster = LocalCluster(
    #     n_workers=4,
    #     threads_per_worker=1,
    #     dashboard_address=":8787",
    # )
    # client = Client(cluster)

    # print("Connected to distributed scheduler")
    # print("Workers/threads:", client.nthreads())
    # print("Total threads:", sum(client.nthreads().values()))
    # info = client.scheduler_info()
    # print("n workers:", len(info["workers"]))
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        # NOTE: creating new model each month, no warm start
        # note: batch_size is within-chunk (using default)
        # print(f'Creating new model (n_clusters: {args.n_clusters})... ({time.time()-t0:.2f})')
        # model = MiniBatchKMeans(n_clusters=args.n_clusters)

        # ----
        # FIT
        # ----
        
        print(f'\nLoading embeddings {year}-{month} ... ({time.time()-t0:.2f})')
        ddata = da.from_zarr(
            os.path.join(embed_path, f'embeddings_{year}-{month}.zarr')
        )
        print(f'> Shape: {ddata.shape} (in {len(ddata.chunks[0])} chunks of {ddata.chunks[0][0]})')

        # persist in memory
        # print(f"> Persisting in memory ...")
        # Xp = ddata.persist()
        # progress(Xp)
        # wait(Xp)
        # del ddata

        # # fit kmeans (dask default is kmeans|| for init)
        # print(f"> Fitting model ... ({time.time()-t0:.2f})")
        # # C, _, index = fit_kmeans_faiss(ddata, args.n_clusters, C0=C0)
        # km = dask_ml.cluster.KMeans(
        #     n_clusters=args.n_clusters, 
        #     init="k-means||",
        #     n_init=2,
        #     init_max_iter=8, 
        #     oversampling_factor=10
        # )
        # km.fit(Xp)

        # print(ts.data)

        print(f"Consolidating data into one array ... ({time.time()-t0:.2f})")
        chunks = []
        i = 0
        for chunk in ddata.to_delayed().ravel():
            print(f"> Chunk {i}  ({time.time()-t0:.2f})")
            arr = chunk.compute()
            chunks.append(arr)
            i += 1
        chunks = np.vstack(chunks)

        km = KMeans(n_clusters=args.n_clusters, algorithm="lloyd")      

        print(f"Clustering... ({time.time()-t0:.2f})")
        km.fit(chunks)
        
        # save just cluster centroids
        cc_name = f'model_cc_{year}-{month}.npz'
        with open(os.path.join(model_path, cc_name), 'wb') as f:
            np.savez_compressed(f, cc=km.cluster_centers_.copy(), allow_pickle=False)

        print(f"COMPLETE ({time.time()-t0:.2f})")

        return
    
        # ----
        # LABEL + FIND top_k TO CENTROID
        # ----
        # (need top_k for tf-idf later, 
        #  so just do while we're doing pass over data for labels)

        print(f'Labeling (size: {L}) ...')

        # this will hold all labels
        labels = []

        # find top_k datapoints by distance to each centroid
        cluster_centers = C.copy()
        closest_heaps = [[] for _ in range(args.n_clusters)]
        
        # global index to where we are in the array
        global_index = 0

        # iterate on every chunk of embedding again
        for chunk in ddata.to_delayed().ravel():
            arr = chunk.compute()
            labels_chunk = predict_kmeans_faiss(arr, C, index=index)

            # For each cluster c, we extract the rows in this chunk, compute distances,
            # and then keep a local top_k. We'll push that into the global top_k heap.
            for c in range(args.n_clusters):
                local_indices = np.where(labels_chunk == c)[0]
                if local_indices.size == 0:
                    continue

                # Calculate distances to centroid c
                dist_c = np.linalg.norm(arr[local_indices] - cluster_centers[c], axis=1)

                # Find the chunk’s top_k rows (or fewer if the cluster has <k here)
                # argpartition better than argsort here, only need top_k, unsorted
                k_local = min(args.top_k, local_indices.size)
                if k_local > 0:
                    top_k_local = np.argpartition(dist_c, k_local-1)[:k_local]
                else:
                    top_k_local = []

                # Insert that local top_k into the cluster's global heap
                for idx in top_k_local:
                    dist_val = dist_c[idx]
                    global_idx_of_point = global_index + local_indices[idx]

                    # If heap isn't filled yet, just push
                    if len(closest_heaps[c]) < args.top_k:
                        heapq.heappush(closest_heaps[c], (-dist_val, global_idx_of_point))
                    else:
                        # If this new distance is better (smaller), replace the worst in the heap
                        worst_neg_dist, _ = closest_heaps[c][0]  # largest distance in top-k
                        if -dist_val > worst_neg_dist:
                            heapq.heapreplace(closest_heaps[c], (-dist_val, global_idx_of_point))

            global_index += arr.shape[0]
            labels.append(labels_chunk)

        # --- Extract final top_k indices (sorted by distance ascending) ---
        closest_indices_per_cluster = {}
        for c in range(args.n_clusters):
            # heap entries are (-distance, global_idx)
            sorted_heap = sorted(closest_heaps[c], key=lambda x: x[0])
            closest_indices_per_cluster[c] = [idx for negdist, idx in sorted_heap]

        labels = np.concatenate(labels)

        # Save labels
        print(f'> Saving labels ... ({time.time()-t0:.2f})')
        label_name = f'labels_{year}-{month}.npz'
        with open(os.path.join(label_path, label_name), 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)

        # ---
        # TFIDF
        # ---

        print(f'Extracting text representations ... ')

        print(f'> Building sample corpus ... ({time.time()-t0:.3f})')
        corpus = [[] for _ in range(args.n_clusters)]
        idx_to_cluster = {ix: c for c in closest_indices_per_cluster \
                          for ix in closest_indices_per_cluster[c]}
        # path to compressed file
        file_path = None
        for ext in ['.bz2', '.xz', '.zst']:
            potential_path = os.path.join(data_path, f'RC_{year}-{month}{ext}')
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        if file_path is None:
            raise FileNotFoundError(
                f"No file found for {year}-{month} with supported extensions (.bz2, .xz, .zst)")
        
        for c, sentence in read_sentences(file_path, idx_to_cluster):
            corpus[c].append(sentence)
        
        # collapse into a format for tf-idf vectorizor
        for i in range(len(corpus)):
            corpus[i] = ' --- '.join(corpus[i])

        print(f'> Computing tf-idf ... ({time.time()-t0:.3f})')
        vectorizer = TfidfVectorizer(
            input='content',
            max_df=args.max_df,
            # max_features=100,
            use_idf=True,
            smooth_idf=True
        )

        X = vectorizer.fit_transform(corpus)

        print(f'> Extracting top {args.top_m} keywords ... ({time.time()-t0:.3f})')
        keywords = []
        for i in range(args.n_clusters):
            max_idx = np.argsort(X[i,:].toarray().flatten())[::-1][:args.top_m]
            keyword = vectorizer.get_feature_names_out()[max_idx]
            keywords.append(keyword)

        print(f'> Saving output ... ({time.time()-t0:.3f})')
        output = {
            'year': year,
            'month': month,
            'full': {
                'scores': X,
                'feature_names': vectorizer.get_feature_names_out()
            },
            'tfidf': {
                i: {
                    'sample_indices': closest_indices_per_cluster[i],
                    'keywords': keywords[i],
                } for i in range(args.n_clusters)
            }
        }

        with open(os.path.join(tfidf_path, f'tfidf_{year}-{month}.pkl'), 'wb') as f:
            pickle.dump(output, f)

        print('Garbage collection ...')
        gc.collect()

    print(f'Complete. ({time.time()-t0:.2f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, required=True)
    parser.add_argument('--end-month', type=int, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--top-m', type=int, default=20)
    parser.add_argument('--max-df', type=float, default=0.3)
    args = parser.parse_args()

    # top_k   num sentences closest to centroid to use
    # top_m   num keywords to store
    # max_df  max doc freq threshold for tfidf to include term
    
    subpath = os.path.join(g['save_path'], args.subpath)
    
    for subdir in ['labels', 'models', 'tfidf']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir), exist_ok=True)
    
    train_cluster_model(
        args,
        data_path=g['data_path'],
        embed_path=g['embed_path'],
        label_path=os.path.join(subpath, 'labels'),
        model_path=os.path.join(subpath, 'models'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
    )
        