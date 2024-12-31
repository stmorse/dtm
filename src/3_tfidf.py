"""
Loads clusters labels and sentences
Trains tfidf on every cluster
Saves to file
"""

import configparser
import gc
import json
import os
import pickle
import time

import bz2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def train_tfidf(
    data_path: str,
    embed_path: str,
    model_path: str,
    label_path: str,
    tfidf_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {os.cpu_count()}')
    print(f'tf-idf range              : {years}, {months}')
    print(f'Saving tf-idf to path     : {tfidf_path}\n')

    # number of embeddings/sentences closest to centroid
    # more allows a better tf-idf estimate of keywords
    top_k = 100

    # number of keywords to store
    top_m = 10

    # each month structured as:
    # {
    #   year:
    #   month: 
    #   tfidf:
    #       {cluster #: 
    #           {
    #               sample_indices: [], (top_k)
    #               keywords: [],       (top_m)
    #           }}}}

    for year in years:
        print(f'\nProcessing {year} ... ({time.time()-t0:.3f})')

        for month in months:
            print(f'> Processing month {month} ... ({time.time()-t0:.3f})')

            # load cluster centers
            print(f'   Loading model ... ({time.time()-t0:.3f})')
            with open(os.path.join(model_path, f'model_cc_{year}-{month}.npz'), 'rb') as f:
                model_cc = np.load(f)['cc']
            num_clusters = model_cc.shape[0]

            # load embeddings
            print(f'   Loading embeddings ... ({time.time()-t0:.3f})')
            with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
                embeddings = np.load(f)['embeddings']

            # load labels
            print(f'   Loading labels ... ({time.time()-t0:.3f})')
            with open(os.path.join(label_path, f'labels_{year}-{month}.npz'), 'rb') as f:
                labels = np.load(f)['labels']

            # get index of top_k closest to centroid
            print(f'   Finding top {top_k} closest indices ... ({time.time()-t0:.3f})')
            closest_idx = np.zeros((num_clusters, top_k))
            for i in range(num_clusters):
                query_vector = model_cc[i]
                label_idx = np.where(labels == i)[0]
                cluster_embeddings = embeddings[label_idx]
                distances = np.linalg.norm(cluster_embeddings - query_vector, axis=1)
                closest_points = np.argsort(distances)[:top_k]
                closest_idx[i,:] = closest_points

            # TODO: this is hacky. don't need to load all sentences to pull some samples
            # load sentences
            print(f'   Loading sentences ... ({time.time()-t0:.3f})')
            sentences = []
            with bz2.BZ2File(os.path.join(data_path, f'RC_{year}-{month}.bz2'), 'rb') as f:
                for line in f:
                    entry = json.loads(line)
                    if 'body' not in entry or entry['author']=='[deleted]':
                        continue
                    sentences.append(entry['body'])

            # build "corpus" aka sample sentences near centroids
            # (row=cluster, entry=concat'd sentences)
            print(f'   Building corpus ... ({time.time()-t0:.3f})')
            corpus = ['' for _ in range(num_clusters)]
            for i in range(num_clusters):
                corpus[i] += ' --- '.join([sentences[int(j)] for j in closest_idx[i,:]])

            print(f'   Computing tf-idf ... ({time.time()-t0:.3f})')
            vectorizer = TfidfVectorizer(
                input='content',
                max_df=0.3,
                # max_features=100,  # this seems to get rid of weird/rare words
                use_idf=True,
                smooth_idf=True
            )

            X = vectorizer.fit_transform(corpus)

            print(f'   Extracting top {top_m} keywords ... ({time.time()-t0:.3f})')
            keywords = []
            for i in range(num_clusters):
                max_idx = np.argsort(X[i,:].toarray().flatten())[::-1][:top_m]
                keyword = vectorizer.get_feature_names_out()[max_idx]
                keywords.append(keyword)

            print(f'   Saving output ... ({time.time()-t0:.3f})')
            output = {
                'year': year,
                'month': month,
                'tfidf': {
                    i: {
                        'sample_indices': closest_idx[i,:],
                        'keywords': keywords[i],
                    } for i in range(num_clusters)
                }
            }

            with open(os.path.join(tfidf_path, f'tfidf_{year}-{month}.pkl'), 'wb') as f:
                pickle.dump(output, f)

            print(f'   Garbage collection ... ({time.time()-t0:.3f})')
            del sentences
            del embeddings
            gc.collect()

        # end month
    # end year

    print(f'Complete. ({time.time()-t0:.3f})')


if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    c = config['tfidf']

    train_tfidf(
        data_path=config['general']['data_path'],
        embed_path=os.path.join(config['general']['save_path'], c['embed_subpath']),
        model_path=os.path.join(config['general']['save_path'], c['model_subpath']),
        label_path=os.path.join(config['general']['save_path'], c['label_subpath']),
        tfidf_path=os.path.join(config['general']['save_path'], c['tfidf_subpath']),
        start_year=int(c['start_year']),
        end_year=int(c['end_year']),
        start_month=int(c['start_month']),
        end_month=int(c['end_month']),
    )
