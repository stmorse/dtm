# this prevents FutureWarning's coming from a sklearn dependency
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import configparser
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
# import umap
import dask.array as da
from sklearn.decomposition import PCA

from utils import manova_pillai_trace


def manova(
    embed_path: str,
    align_path: str,
    tfidf_path: str,
    start_year: int,
    end_year: int,
    min_group_size: int,      # min topic centroids in group    
):
    print('Loading alignment model ...')

    with open(os.path.join(align_path, 'max_100/align_model_HDBSCAN.pkl'), 'rb') as f:
        align_model = pickle.load(f)

    # number of clusters (and therefore number centroids in each time_period)
    # TODO: don't really want this hardcoded
    Ck = 50

    # unique_group_labels, counts = np.unique(align_model.labels_, return_counts=True)
    
    # trim this group a bit
    labs, cnts = np.unique(align_model.labels_, return_counts=True)
    unique_group_labels = np.array([
        lab for lab, cnt in zip(labs, cnts) \
        if lab != -1 and cnt >= min_group_size
    ])

    print(f'Testing {len(unique_group_labels)} groups')

    yrmo = [(yr, mo) for yr in range(start_year, end_year+1) for mo in range(1,13)]
    print(f'Total time periods: ', len(yrmo))

    # for each group idx, all_samples_dict[goi] will be ndarray with 
    # 100 samples per member topic cluster
    # all_samples_labels[goi] will be corresponding topic cluster labels
    all_samples_dict = {goi: [] for goi in unique_group_labels}
    all_samples_labels = {goi: [] for goi in unique_group_labels}

    # iterate on time period
    for i, (yr, mo) in enumerate(yrmo):
        print(f'\nTIME PERIOD: {yr}-{mo:02}')

        # get tfidf for this time period
        # contains top vectors for each topic cluster
        with open(os.path.join(tfidf_path, f'tfidf_{yr}-{mo:02}.pkl'), 'rb') as f:
            tfidf = pickle.load(f)['tfidf']

        # slice out group labels for this time period
        # this is a Ck-long list of actual group labels
        ga, gb = i * Ck, (i + 1) * Ck
        window_group_labels = align_model.labels_[ga:gb]
        
        # extract all embeddings for this time period
        print('Extract embeddings ... ')
        
        # figure out if embeddings are .npz or .zarr
        filepath, ex = None, None
        for ext in ['.zarr', '.npz']:
            potential_path = os.path.join(embed_path, f'embeddings_{yr}-{mo:02}{ext}')
            if os.path.exists(potential_path):
                filepath = potential_path
                ex = ext
                break
        if filepath is None:
            raise FileNotFoundError(
                f'No file found for {yr}-{mo} with .zarr or .npz'
            )
            return

        # if npz, load the whole thing and extract indices for each goi (easy)
        # if zarr, loop over each chunk and append indices for each goi
                
        embeddings = []

        if ex=='.npz':
            print('> Processing (single) file ... ')

            with open(filepath, 'rb') as f:
                embeddings = np.load(f)['embeddings']

            for goi in unique_group_labels:
                # cluster labels corresponding to clusters of this group (goi) for this time period
                # ex. [4, 6] means cluster 4 and 6 for this time period
                # because in each window, group_labels index is 0-49
                cluster_labels = np.where(window_group_labels == goi)[0]

                # possible there are no clusters for this group in this time period
                if len(cluster_labels)==0:
                    continue

                # loop each cluster label and append top embeddings to samples dict
                for cluster_label in cluster_labels:
                    top_ix = np.array(tfidf[cluster_label]['sample_indices']).astype(int)
                    all_samples_dict[goi].append(
                        embeddings[top_ix, :]
                    )
                    all_samples_labels[goi].append(
                        [cluster_label] * len(top_ix)
                    )
            
        # data too big to hold all embeddings for the time period in memory
        # so we extract top indices within each chunk
        # this involves multiple passes over tfidf and group labels (fast)
        # but one pass over embeddings (slow)
        elif ex=='.zarr':
            print('> Processing chunked file ... ')

            ddata = da.from_zarr(filepath)
            
            for j, chunk in enumerate(ddata.to_delayed().ravel()):
                arr = chunk.compute()
                chunk_size = arr.shape[0]

                print(f'> Chunk {j} (size {chunk_size}) ...')
                
                for goi in unique_group_labels:
                    cluster_labels = np.where(window_group_labels == goi)[0]

                    if len(cluster_labels) == 0:
                        continue

                    for cluster_label in cluster_labels:
                        # these are window wide indices
                        top_ix = np.array(tfidf[cluster_label]['sample_indices']).astype(int)

                        # now get chunk level indices
                        ca, cb = j * chunk_size, (j+1) * chunk_size
                        top_zix = np.where((top_ix >= ca) & (top_ix < cb))[0]
                        top_cix = top_ix[top_zix]

                        # now append to samples dict
                        all_samples_dict[goi].append(
                            arr[top_cix, :]
                        )
                        all_samples_labels[goi].append(
                            [cluster_label] * len(top_cix)
                        )

    # end time period loop

    # now all_samples_dict and all_samples_labels has everything we need
    # need to concat into single ndarray's per group
    # and project down
    # u_embedder = umap.UMAP(
    #         n_neighbors=15,
    #         n_components=10,
    #         metric='euclidean',
    #         init='spectral',
    #         min_dist=0.1,
    #         spread=1.0
    #     )
    
    print(f'Preparing all samples ... ')
    for goi in unique_group_labels:
        print(f'> Processing group {goi} ... ')
        all_samples_dict[goi] = np.vstack(all_samples_dict[goi])
        all_samples_labels[goi] = np.concatenate(all_samples_labels[goi]).astype(int)

        # all_samples_dict[goi] = u_embedder.fit_transform(
        #     all_samples_dict[goi]
        # )
        all_samples_dict[goi] = PCA(n_components=10).fit_transform(
            all_samples_dict[goi]
        )

    print(f'Computing MANOVA ... ')
    for goi in unique_group_labels:
        # compute manova
        print(f'> Group {goi}: ...')
        pillai, F_stat, p_val = manova_pillai_trace(
            all_samples_dict[goi], 
            all_samples_labels[goi]
        )

        print(f'   Pillais Trace: {pillai:.4f}')
        print(f'   Approx F-Statistic: {F_stat:.4f}')
        print(f'   p-value: {p_val:.6f}')


if __name__=="__main__":
    os.environ['PYTHONUNBUFFERED'] = '1'

    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--min-group-size', type=int, required=True)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)
    
    manova(
        embed_path=g['embed_path'],
        align_path=os.path.join(subpath, 'align'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_group_size=int(args.min_group_size),
    )