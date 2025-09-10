import argparse
import configparser
import json
import os
import time

import numpy as np
import joblib
import dask.array as da

def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--special', type=str, default="")
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, required=True)
    parser.add_argument('--end-month', type=int, required=True)
    args = parser.parse_args()
    
    subpath = os.path.join(g['save_path'], args.subpath)
    if len(args.special) > 0:
        subpath = os.path.join(subpath, args.special)
    
    # augment args with paths
    setattr(args, "embed_path", g["embed_path"])
    setattr(args, "model_path", os.path.join(subpath, 'models'))
    setattr(args, "label_path", os.path.join(subpath, 'labels'))

    # start time (used in _log)
    global t0
    t0 = time.time()

    # compute inertia
    compute_inertia(args)

def _log(msg):
    t = time.time() - t0
    print(f"{msg} ... ({t:.2f})")

def compute_inertia(args):
    t0 = time.time()
    years = [str(y) for y in range(args.start_year, args.end_year+1)]
    months = [f'{m:02}' for m in range(args.start_month, args.end_month+1)]

    print(f'CPU count                 : {joblib.cpu_count()}, {os.cpu_count()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving results to path    : {args.model_path}\n')

    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f"\n{'='*40}\n{year}-{month}\n")

        fname = f'embeddings_{year}-{month}'
        fpath = os.path.join(args.embed_path, f'{fname}.zarr')

        # -- LOAD EMBEDDINGS --
        ddata = da.from_zarr(fpath)
        L = ddata.shape[0]
        M = len(ddata.chunks[0])
        _log(f'> Shape: {L} (in {M} chunks of {ddata.chunks[0][0]})')

        # -- LOAD LABELS --
        lname = f"labels_{year}-{month}"
        lpath = os.path.join(args.label_path, f"{lname}.npz")
        with open(lpath, "rb") as f:
            labels = np.load(f)["labels"]

        # -- LOAD MODEL (CENTROIDS) --
        cname = f"model_cc_{year}-{month}"
        cpath = os.path.join(args.model_path, f"{cname}.npz")
        with open(cpath, "rb") as f:
            cc = np.load(f)["cc"]

        # -- COMPUTE WITHIN-CLUSTER SUM OF SQUARES --
        i = 0
        wcss = 0.0
        global_index = 0
        for chunk in ddata.to_delayed().ravel():
            _log(f"  - Chunk {i}")
            i += 1

            # compute chunk to numpy array
            arr = chunk.compute()
            chunk_size = arr.shape[0]

            # get label slice for this chunk
            labels_chunk = labels[global_index:global_index + chunk_size]

            # for each cluster, compute sum of squared distances 
            # for points in this chunk
            for c in range(cc.shape[0]):
                mask = (labels_chunk == c)

                # check we don't have all 0's 
                # (no points in this cluster in this chunk)
                if np.any(mask):  
                    points = arr[mask]
                    center = cc[c]
                    wcss += np.sum((points - center) ** 2)

            global_index += chunk_size

        _log(f"WCSS for {year}-{month}: {wcss}")
        
        # -- SAVE --
        wcss_path = os.path.join(args.model_path, f"wcss_{year}-{month}.json")
        stats = {"wcss": wcss, "avg_wcss": wcss / L}
        with open(wcss_path, "w") as f:
            json.dump(stats, f)
        _log(f"(Avg WCSS: {wcss / L})")

    _log("COMPLETE")

if __name__ == "__main__":
    main()

            