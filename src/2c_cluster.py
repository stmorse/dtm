import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import os
import pickle
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import dask.array as da

from utils import read_sentences

def main():
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
    parser.add_argument('--model', type=str, default="mbkm")
    parser.add_argument('--chunk_size', type=int, default=1_000_000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--top-m', type=int, default=20)
    parser.add_argument('--max-df', type=float, default=0.3)
    args = parser.parse_args()

    # NOTES:
    # chunk_size only used for NPZ arrays
    # top_k   num sentences closest to centroid to use
    # top_m   num keywords to store
    # max_df  max doc freq threshold for tfidf to include term
    
    subpath = os.path.join(g['save_path'], args.subpath)
    
    # ensure directories exist
    for subdir in ['labels', 'models', 'tfidf']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir), exist_ok=True)

    # augment args with paths
    setattr(args, "data_path", g["data_path"])
    setattr(args, "embed_path", g["embed_path"])
    setattr(args, "model_path", os.path.join(subpath, 'models'))
    setattr(args, "label_path", os.path.join(subpath, 'labels'))
    setattr(args, "tfidf_path", os.path.join(subpath, 'tfidf'))

    # start time (used in _log)
    global t0
    t0 = time.time()

    # begin cluster
    cluster(args)

def _log(msg):
    t = time.time() - t0
    print(f"{msg} ... ({t:.2f})")

def cluster(args):
    t0 = time.time()
    years = [str(y) for y in range(args.start_year, args.end_year+1)]
    months = [f'{m:02}' for m in range(args.start_month, args.end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {args.label_path}\n')

    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f"\n{'='*40}\n{year}-{month}\n")

        # create new model each month
        _log(f'Creating model: {args.model}, n_clusters: {args.n_clusters}')
        if args.model == "mbkm":
            model = MiniBatchKMeans(n_clusters=args.n_clusters)
        elif args.model == "km":
            model = KMeans(n_clusters=args.n_clusters, algorithm="lloyd")
        else:
            raise ValueError(f"Model not recognized ({args.model}).")

        # ----
        # FIT
        # ----
                
        fname = f'embeddings_{year}-{month}'
        zarr_path = os.path.join(args.embed_path, f'{fname}.zarr')
        npz_path = os.path.join(args.embed_path, f'{fname}.npz')
        ftype = None

        # zarr is preferred, check first
        if os.path.exists(zarr_path):
            fpath = zarr_path
            ftype = "zarr"
        elif os.path.exists(npz_path):
            fpath = npz_path
            ftype = "npz"
        else:
            raise FileNotFoundError(f'No embedding found (.npz or .zarr).')
        
        _log(f'Loading embeddings {year}-{month} ({ftype})')
        
        # --- ZARR ---
        if ftype == "zarr":
            
            # -- LOAD --
            ddata = da.from_zarr(fpath)
            L = ddata.shape[0]
            M = len(ddata.chunks[0])
            CS = ddata.chunks[0][0]
            _log(f'> Shape: {L} (in {M} chunks of {ddata.chunks[0][0]})')

            # -- FIT MBKM --
            if args.model == "mbkm":
                i = 0
                for chunk in ddata.to_delayed().ravel():
                    arr = chunk.compute()  
                    _log(f'> Fitting chunk {i+1}/{M} ({arr.shape[0]})')
                    model.partial_fit(arr)
                    i += 1

            # -- FIT KM --
            elif args.model == "km":
                _log(f"> Consolidating")
                embeddings = []
                i = 0
                for chunk in ddata.to_delayed().ravel():
                    _log(f"  - Chunk {i}")
                    arr = chunk.compute()
                    embeddings.append(arr)
                    i += 1
                embeddings = np.vstack(embeddings)

                _log(f"> Clustering (KM)")
                model.fit(embeddings)

        # --- NPZ ---
        elif ftype == "npz":

            # -- LOAD --
            with open(fpath, 'rb') as f:
                embeddings = np.load(f)['embeddings']
            L = len(embeddings)
            M = L // args.chunk_size
            CS = args.chunk_size
            _log(f"> Shape {L} embeddings in {M} chunks of {CS}")

            # -- FIT MBKM --
            if args.model == "mbkm":
                for i in range(0, M):
                    j = i * CS
                    chunk = embeddings[j:j + CS]

                    _log(f'> Fitting chunk {i}/{M} ({len(chunk)})')
                    model.partial_fit(chunk)
                    # del chunk
                    # gc.collect()

                # fit on "leftovers"
                fidx = M * CS
                if fidx < L:
                    leftovers = embeddings[fidx:]
                    model.partial_fit(leftovers)

            # -- FIT KM --
            elif args.model == "km":
                _log(f"> Clustering (KM)")
                model.fit(embeddings)

        # --- SAVE ---

        # save just cluster centroids
        cc_path = os.path.join(args.model_path, f'model_cc_{year}-{month}.npz')
        _log(f"Saving ({cc_path})")
        with open(cc_path, 'wb') as f:
            np.savez_compressed(
                f, 
                cc=model.cluster_centers_.copy(), 
                allow_pickle=False
            )

        # ----
        # LABEL
        # ----

        labels = []

        _log(f"Labeling (size {L}, {M} chunks)")

        # find top_k datapoints by distance to each centroid
        cluster_centers = model.cluster_centers_
        closest_heaps = [[] for _ in range(args.n_clusters)]

        # global index to where we are in the array
        global_index = 0

        # handle all the different data / model arrangements
        # so we can do one loop
        if ftype == "zarr" and args.model == "mbkm":
            # data is in chunks already and we want to handle it that way
            iterator = ddata.to_delayed().ravel()
            compute = True
        else:
            # data is in a single `embeddings` object but 
            # we'll still handle it in chunks

            # M is all chunks for zarr, but want to handle leftovers separately
            # for consistency with npz
            M_ = M if ftype=="npz" else M-1  # M_ is number of "full" chunks
            iterator = range(M_+1)
            compute = False

        i = 0
        for item in iterator:
            _log(f"> Chunk {i+1}/{M}")
            i += 1

            if compute:
                # item is a chunk
                arr = item.compute()
            else:
                # item is an index to a chunk
                if item < M_:
                    j = item * CS
                    arr = embeddings[j:j + CS]
                else:
                    fidx = M_ * CS
                    arr = embeddings[fidx:]

            labels_chunk = model.predict(arr)

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
        lpath = os.path.join(args.label_path, f'labels_{year}-{month}.npz')
        _log(f'> Saving labels ({lpath})')
        with open(lpath, 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)

        # ---
        # TFIDF
        # ---

        _log(f'Extracting text representations ... ')

        _log(f'> Building sample corpus')
        corpus = [[] for _ in range(args.n_clusters)]
        idx_to_cluster = {ix: c for c in closest_indices_per_cluster \
                          for ix in closest_indices_per_cluster[c]}
        # path to compressed file
        file_path = None
        for ext in ['.bz2', '.xz', '.zst']:
            potential_path = os.path.join(args.data_path, f'RC_{year}-{month}{ext}')
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        if file_path is None:
            raise FileNotFoundError(f"No {year}-{month} with (.bz2, .xz, .zst)")
        
        for c, sentence in read_sentences(file_path, idx_to_cluster):
            corpus[c].append(sentence)

        # collapse into a format for tf-idf vectorizor
        for i in range(len(corpus)):
            corpus[i] = ' --- '.join(corpus[i])

        _log(f'> Computing tf-idf')
        vectorizer = TfidfVectorizer(
            input='content',
            max_df=args.max_df,
            # max_features=100,
            use_idf=True,
            smooth_idf=True
        )

        X = vectorizer.fit_transform(corpus)

        _log(f'> Extracting top {args.top_m} keywords')
        keywords = []
        for i in range(args.n_clusters):
            max_idx = np.argsort(X[i,:].toarray().flatten())[::-1][:args.top_m]
            keyword = vectorizer.get_feature_names_out()[max_idx]
            keywords.append(keyword)

        tfidf_path = os.path.join(args.tfidf_path, f'tfidf_{year}-{month}.pkl')
        _log(f'> Saving output ({tfidf_path})')
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
        with open(tfidf_path, 'wb') as f:
            pickle.dump(output, f)

        # ---
        # SCORE
        # ---

        avg_inertia = model.inertia_ / L
        stats = {
            "inertia": model.inertia_, "avg_inertia": avg_inertia,
            "num_chunks": M, "size": L
        }
        ipath = os.path.join(args.model_path, f"stats_{year}-{month}.pkl")
        _log(f"Model inertia: {model.inertia_:.3f} (Avg: {avg_inertia})")
        with open(ipath, "wb") as f:
            pickle.dump(stats, f)

        _log('Garbage collection')
        gc.collect()

    _log("\n\nCOMPLETE\n")

if __name__ == "__main__":
    main()