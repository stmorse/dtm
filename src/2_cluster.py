"""
Loads embeddings into memory
Trains cluster model on every month
Labels that month
Saves to file

Trains tfidf on every cluster
Saves to file

output = {
    'year': year,
    'month': month,
    'full': {
        'scores': np.ndarray (TfidfVectorizer result),
        'feature_names': vectorizer.get_feature_names_out()
    },
    'tfidf': {
        i: {
            'sample_indices': closest_idx[i,:],
            'keywords': keywords[i],
        } for i in range(num_clusters)
    }
}
"""

import bz2
import configparser
import gc
import json
import os
import pickle
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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
    chunk_size: int,
    top_k: int,         # num sentences closest to centroid to use
    top_m: int,         # num keywords to store
    max_df: float,      # max doc freq threshold for tfidf to include term
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving labels to path     : {label_path}')
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'\nLoading embeddings {year}-{month} ... ({time.time()-t0:.2f})')
        with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
            embeddings = np.load(f)['embeddings']
            
        # ----
        # FIT
        # ----

        # NOTE: creating new model each month, no warm start
        print(f'Creating new model (n_clusters: {n_clusters})... ({time.time()-t0:.2f})')
        # we're using partial_fit on chunks so I think batch_size is irrelevant
        model = MiniBatchKMeans(n_clusters=n_clusters)

        # partial_fit over all chunks
        L = len(embeddings)
        M = L // chunk_size   # num chunks
        print(f'Fitting model (size: {L}) (chunks: {M+1}) ...')
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

        print(f'> Cluster model saved to {model_path} ({time.time()-t0:.2f})')

        # ----
        # LABEL
        # ----

        labels = []
        print(f'Labeling (size: {L}) (chunks: {M+1}) ...')
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

        # Save labels
        print(f'> Saving labels ... ({time.time()-t0:.2f})')
        label_name = f'labels_{year}-{month}.npz'
        with open(os.path.join(label_path, label_name), 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)

        # ---
        # TFIDF
        # ---

        print(f'Computing tf-idf ... ')

        # get index of top_k closest to centroid
        print(f'> Finding top {top_k} closest indices ... ({time.time()-t0:.3f})')
        model_cc = model.cluster_centers_
        closest_idx = np.zeros((n_clusters, top_k))
        for i in range(n_clusters):
            query_vector = model_cc[i]
            label_idx = np.where(labels == i)[0]
            cluster_embeddings = embeddings[label_idx]
            distances = np.linalg.norm(cluster_embeddings - query_vector, axis=1)
            closest_points = np.argsort(distances)[:top_k]
            # NOTE: closest_points is *within* label so need to convert back
            closest_idx[i,:] = label_idx[closest_points]
        
        # TODO: this is hacky. don't need to load all sentences to pull some samples
        # load sentences
        print(f'> Loading sentences ... ({time.time()-t0:.3f})')
        sentences = []
        with bz2.BZ2File(os.path.join(data_path, f'RC_{year}-{month}.bz2'), 'rb') as f:
            for line in f:
                entry = json.loads(line)
                if 'body' not in entry or entry['author']=='[deleted]':
                    continue
                sentences.append(entry['body'])
        print(f'> (num sentences: {len(sentences)})')

        # build "corpus" aka sample sentences near centroids
        # (row=cluster, entry=concat'd sentences)
        print(f'> Building corpus ... ({time.time()-t0:.3f})')
        corpus = ['' for _ in range(n_clusters)]
        for i in range(n_clusters):
            corpus[i] += ' --- '.join([sentences[int(j)] for j in closest_idx[i,:]])

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
                    'sample_indices': closest_idx[i,:],
                    'keywords': keywords[i],
                } for i in range(n_clusters)
            }
        }

        with open(os.path.join(tfidf_path, f'tfidf_{year}-{month}.pkl'), 'wb') as f:
            pickle.dump(output, f)

        print('Garbage collection ...')
        del sentences
        del embeddings
        gc.collect()

    print(f'Complete. ({time.time()-t0:.2f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    c = config['cluster']

    train_cluster_model(
        data_path=config['general']['data_path'],
        embed_path=os.path.join(g['save_path'], g['embed_subpath']),
        label_path=os.path.join(g['save_path'], g['run_subpath'], 'labels'),
        model_path=os.path.join(g['save_path'], g['run_subpath'], 'models'),
        tfidf_path=os.path.join(g['save_path'], g['run_subpath'], 'tfidf'),
        n_clusters=int(g['n_clusters']),
        start_year=int(g['start_year']),
        end_year=int(g['end_year']),
        start_month=int(g['start_month']),
        end_month=int(g['end_month']),
        chunk_size=int(c['chunk_size']),
        top_k=int(c['top_k']),
        top_m=int(c['top_m']),
        max_df=float(c['max_df'])
    )