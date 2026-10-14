"""
WS3 / R1.2 quick checks on the published mbkm_50 run.

1. Model config: module list (Normalize?), max_seq_length.
2. Stored embedding norms on a few months (zarr and npz).
3. Centroid norms for all 204 months.
4. [removed] / [deleted] artifact clusters. Identical bodies embed to the same
   vector, so every '[removed]' body is a single point in 384-d. For each
   (month, cluster) we record the centroid's distance to that point and the
   tf-idf score of the token 'removed' / 'deleted' among the cluster's top-100
   nearest comments (the tf-idf corpus from 2a_cluster_s.py / 2_cluster.py).

Run from src/:
    python 10a_quickchecks.py --seed 0
"""

import argparse
import configparser
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import zarr

ARTIFACTS = ['[removed]', '[deleted]']


def month_list(save_path, run_subpath):
    files = sorted(glob.glob(os.path.join(save_path, run_subpath, 'models', 'model_cc_*.npz')))
    return [os.path.basename(f)[len('model_cc_'):-len('.npz')] for f in files]


def load_embeddings_sample(embed_path, ym, max_rows):
    """First max_rows rows of a month (zarr or npz)."""
    zp = os.path.join(embed_path, f'embeddings_{ym}.zarr')
    if os.path.exists(zp):
        z = zarr.open(zp, mode='r')
        return np.asarray(z[:max_rows]), 'zarr', z.dtype
    X = np.load(os.path.join(embed_path, f'embeddings_{ym}.npz'))['embeddings']
    return X[:max_rows], 'npz', X.dtype


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    e = config['embed']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--norm_months', type=str, default='2006-06,2008-06,2012-06,2018-06')
    parser.add_argument('--norm_rows', type=int, default=400000)
    parser.add_argument('--out_path', type=str, default='../output/ws3/10a')
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    t0 = time.time()
    res = {'params': vars(args)}

    # ---- 1. model config ----
    os.environ.setdefault('HF_HOME', g['hf_path'])
    import torch
    from sentence_transformers import SentenceTransformer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # same settings as 1_embed.py
    model = SentenceTransformer(e['hf_model'], device=dev,
                                model_kwargs={'torch_dtype': 'float16'})
    res['model'] = {
        'name': e['hf_model'],
        'modules': [type(m).__name__ for m in model],
        'max_seq_length': model.max_seq_length,
        'tokenizer_model_max_length': model.tokenizer.model_max_length,
        'dim': model.get_sentence_embedding_dimension(),
    }
    print(res['model'])

    art = model.encode(ARTIFACTS, convert_to_numpy=True).astype(np.float64)

    # ---- 2. embedding norms ----
    res['embedding_norms'] = {}
    for ym in args.norm_months.split(','):
        X, fmt, dt = load_embeddings_sample(g['embed_path'], ym, args.norm_rows)
        n = np.linalg.norm(X.astype(np.float64), axis=1)
        res['embedding_norms'][ym] = {
            'format': fmt, 'dtype': str(dt), 'rows_checked': int(len(n)),
            'min': float(n.min()), 'max': float(n.max()), 'mean': float(n.mean()),
            'n_abs_dev_gt_1e-3': int((np.abs(n - 1) > 1e-3).sum()),
        }
        # does the '[removed]' vector appear verbatim in the stored rows?
        for a, v in zip(ARTIFACTS, art):
            d = np.linalg.norm(X.astype(np.float64) - v, axis=1)
            res['embedding_norms'][ym][f'frac_rows_eq_{a}'] = float((d < 1e-2).mean())
        print(ym, res['embedding_norms'][ym], f'({time.time()-t0:.1f}s)')

    # ---- 3 + 4. centroids, artifact clusters ----
    run = os.path.join(g['save_path'], args.run_subpath)
    rows = []
    for ym in month_list(g['save_path'], args.run_subpath):
        cc = np.load(os.path.join(run, 'models', f'model_cc_{ym}.npz'))['cc'].astype(np.float64)
        lab = np.load(os.path.join(run, 'labels', f'labels_{ym}.npz'))['labels']
        counts = np.bincount(lab, minlength=cc.shape[0])
        with open(os.path.join(run, 'tfidf', f'tfidf_{ym}.pkl'), 'rb') as f:
            tf = pickle.load(f)
        fn = list(tf['full']['feature_names'])
        sc = tf['full']['scores']
        tfcol = {w: (sc[:, fn.index(w)].toarray().ravel() if w in fn else np.zeros(cc.shape[0]))
                 for w in ['removed', 'deleted']}
        for c in range(cc.shape[0]):
            rows.append({
                'month': ym, 'cluster': c, 'row': None, 'n': int(counts[c]),
                'frac_month': counts[c] / counts.sum(),
                'centroid_norm': np.linalg.norm(cc[c]),
                'dist_removed': np.linalg.norm(cc[c] - art[0]),
                'dist_deleted': np.linalg.norm(cc[c] - art[1]),
                'tfidf_removed': tfcol['removed'][c],
                'tfidf_deleted': tfcol['deleted'][c],
                'top_keywords': ' '.join(tf['tfidf'][c]['keywords'][:8]),
            })
    df = pd.DataFrame(rows)
    months = sorted(df.month.unique())
    df['row'] = df.month.map({m: i for i, m in enumerate(months)}) * 50 + df.cluster
    df.to_csv(os.path.join(args.out_path, 'clusters.csv'), index=False)

    res['centroid_norms'] = df.centroid_norm.describe().to_dict()
    res['artifact_vector_dist_removed_deleted'] = float(np.linalg.norm(art[0] - art[1]))
    with open(os.path.join(args.out_path, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f'done ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
