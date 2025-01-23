import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import os
import pickle
import time

import bz2
import lzma
import zstandard as zstd

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import dask.array as da

def open_compressed(file_path):
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(file_path, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.open(file_path, 'rb')
    elif file_path.endswith('.zst'):
        # For .zst, return a stream reader
        f = open(file_path, 'rb')  # Open file in binary mode
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(f)
    else:
        raise ValueError('Unsupported file extension.')

def read_sentences(file_path, idx_to_cluster):
    """
    Read JSON entries from a compressed file, extract the 'body' field,
    and yield if in the idx_to_cluster dict
    """
    byte_buffer = b""  # For handling partial lines in `.zst` files

    with open_compressed(file_path) as f:
        # Iterate over the file
        i = 0
        for chunk in iter(lambda: f.read(8192), b""):  # Read file in binary chunks
            byte_buffer += chunk

            # Process each line in the byte buffer
            while b"\n" in byte_buffer:
                line, byte_buffer = byte_buffer.split(b"\n", 1)

                # Parse JSON and process the 'body' field
                entry = json.loads(line.decode("utf-8"))

                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue

                # Truncate long 'body' fields
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]

                if i in idx_to_cluster:
                    # we have a candidate
                    c = idx_to_cluster[i]
                    yield c, body

                # increment i whether we yielded or not
                i += 1

    # don't bother with leftovers for corpus building


def train_cluster_model(
    data_path: str,
    embed_path: str,
    label_path: str,
    model_path: str,
    tfidf_path: str,
    n_clusters: int,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    top_k: int,         # num sentences closest to centroid to use
    top_m: int,         # num keywords to store
    max_df: float,      # max doc freq threshold for tfidf to include term
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {label_path}\n')
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        # NOTE: creating new model each month, no warm start
        # note: batch_size is within-chunk (using default)
        print(f'Creating new model (n_clusters: {n_clusters})... ({time.time()-t0:.2f})')
        model = MiniBatchKMeans(n_clusters=n_clusters)

        # ----
        # FIT
        # ----
        
        print(f'\nLoading embeddings {year}-{month} ... ({time.time()-t0:.2f})')
        ddata = da.from_zarr(os.path.join(embed_path, f'embeddings_{year}-{month}.zarr'))
        L = ddata.shape[0]
        print(f'Total {L}.  Chunks: {ddata.chunks}')

        # We'll iterate chunks and do partial fit
        i = 0
        for chunk in ddata.to_delayed().ravel():
            arr = chunk.compute()  
            print(f'> Fitting chunk {i} ({arr.shape[0]}) ... ({time.time()-t0:.2f})')
            model.partial_fit(arr)
            i += 1

        # save just cluster centroids (for joblib incompatibility)
        cc_name = f'model_cc_{year}-{month}.npz'
        with open(os.path.join(model_path, cc_name), 'wb') as f:
            np.savez_compressed(f, cc=model.cluster_centers_.copy(), allow_pickle=False)

        # ----
        # LABEL + FIND top_k TO CENTROID
        # ----
        # (need top_k for tf-idf later, 
        #  so just do while we're doing pass over data for labels)

        print(f'Labeling (size: {L}) ...')

        # this will hold all labels
        labels = []

        # find top_k datapoints by distance to each centroid
        cluster_centers = model.cluster_centers_
        closest_heaps = [[] for _ in range(n_clusters)]
        
        # global index to where we are in the array
        global_index = 0

        # iterate on every chunk of embedding again
        for chunk in ddata.to_delayed().ravel():
            arr = chunk.compute()
            labels_chunk = model.predict(arr)

            # For each cluster c, we extract the rows in this chunk, compute distances,
            # and then keep a local top_k. We'll push that into the global top_k heap.
            for c in range(n_clusters):
                local_indices = np.where(labels_chunk == c)[0]
                if local_indices.size == 0:
                    continue

                # Calculate distances to centroid c
                dist_c = np.linalg.norm(arr[local_indices] - cluster_centers[c], axis=1)

                # Find the chunk’s top_k rows (or fewer if the cluster has <k here)
                # argpartition better than argsort here, only need top_k, unsorted
                k_local = min(top_k, local_indices.size)
                if k_local > 0:
                    top_k_local = np.argpartition(dist_c, k_local-1)[:k_local]
                else:
                    top_k_local = []

                # Insert that local top_k into the cluster's global heap
                for idx in top_k_local:
                    dist_val = dist_c[idx]
                    global_idx_of_point = global_index + local_indices[idx]

                    # If heap isn't filled yet, just push
                    if len(closest_heaps[c]) < top_k:
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
        for c in range(n_clusters):
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
        corpus = [[] for _ in range(n_clusters)]
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
        
        # with bz2.BZ2File(os.path.join(data_path, f'RC_{year}-{month}.bz2'), 'rb') as f:
        #     i = 0
        #     for line in f:
        #         entry = json.loads(line)
        #         if 'body' not in entry or entry['author']=='[deleted]':
        #             continue

        #         # check if i is in top_k of any cluster
        #         if i in idx_to_cluster:
        #             # we have a candidate
        #             c = idx_to_cluster[i]
        #             corpus[c].append(entry['body'])

        #         # increment i whether we added or not
        #         i += 1
        
        # collapse into a format for tf-idf vectorizor
        for i in range(len(corpus)):
            corpus[i] = ' --- '.join(corpus[i])

        print(f'> Computing tf-idf ... ({time.time()-t0:.3f})')
        vectorizer = TfidfVectorizer(
            input='content',
            max_df=max_df,
            # max_features=100,
            use_idf=True,
            smooth_idf=True
        )

        X = vectorizer.fit_transform(corpus)

        print(f'> Extracting top {top_m} keywords ... ({time.time()-t0:.3f})')
        keywords = []
        for i in range(n_clusters):
            max_idx = np.argsort(X[i,:].toarray().flatten())[::-1][:top_m]
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
                } for i in range(n_clusters)
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
    parser.add_argument('--top-k', type=int, required=True)
    parser.add_argument('--top-m', type=int, required=True)
    parser.add_argument('--max-df', type=float, required=True)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)
    
    for subdir in ['labels', 'models', 'tfidf']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir))
    
    train_cluster_model(
        data_path=g['data_path'],
        embed_path=g['embed_path'],
        label_path=os.path.join(subpath, 'labels'),
        model_path=os.path.join(subpath, 'models'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
        n_clusters=args.n_clusters,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        top_k=args.top_k,
        top_m=args.top_m,
        max_df=args.max_df
    )
        