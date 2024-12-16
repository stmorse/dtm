"""
Streams entirety of a month into CPU Mem, 
then batches into multiple GPU for encoding,
and saves to file
"""

import bz2
import json
import time
import pickle
import gc
import configparser
import argparse
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def main(
    data_path: str,
    embed_path: str,
    meta_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    batch_size: int,
    show_progress: bool,
    hf_model: str,
    hf_path: str
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    os.environ['HF_HOME'] = hf_path

    print(f'GPU enabled?             : {torch.cuda.is_available()}')
    print(f'Embedding range          : {years}, {months}')
    print(f'Saving embeddings to path: {embed_path}')
    print(f'Saving metadata to path  : {meta_path}\n')
    print(f'Loading model ... ({time.time()-t0:.2f})')

    model = SentenceTransformer(hf_model,
                                device='cuda',
                                model_kwargs={'torch_dtype': 'float16'})  

    print(f'Starting multiprocessing pool ... ({time.time()-t0:.2f})')  

    pool = model.start_multi_process_pool()
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Processing {year}-{month} ... ({time.time()-t0:.2f})')

        # path to bz2 compressed file
        file_path = f'{data_path}RC_{year}-{month}.bz2'

        # initialize an array to store embeddings
        embeddings = []

        # will hold metadata
        df = []

        # Open the bz2 compressed file
        print(f'> Loading comments (Batch size: {batch_size})... ({time.time()-t0:.2f})')
        j, k = 0, 0   # line count, batch count
        with bz2.open(file_path, 'rb') as f:
            batch = []
            for line in f:
                entry = json.loads(line)
                j += 1

                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue
                
                # quick and dirty to keep entries closer to SBERT token limit (256)
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]

                batch.append(body)

                # continue building metadata
                df.append([k, entry['author'], entry['id'], entry['created_utc']])
                        
                # when the batch is full, process it
                if len(batch) == batch_size:
                    print(f'> Processing batch {k} ({len(batch)})... ({time.time()-t0:.2f})')

                    # encode the batch of sentences on the GPU
                    batch_embeddings = model.encode_multi_process(
                        batch, pool, show_progress_bar=show_progress)
                    embeddings.append(batch_embeddings)
                    
                    batch = []  # Clear batch after processing
                    k += 1
            
            # Process any remaining sentences in the final batch
            if len(batch) > 0:
                print(f'> Leftovers ({len(batch)})... ({time.time()-t0:.2f})')

                batch_embeddings = model.encode_multi_process(
                        batch, pool, show_progress_bar=show_progress)
                
                embeddings.append(batch_embeddings)
                batch = []
                k += 1

        # Convert list of arrays into a single array
        embeddings = np.vstack(embeddings)

        # Save the embeddings to disk
        print(f'> Total lines: {j}  Total embeddings: {len(embeddings)}  Total batches: {k}')
        print(f'> Saving to disk ... ({time.time()-t0:.2f})')
        with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'wb') as f:
            np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

        # save metadata to disk
        print(f'> Saving metadata to disk ... ({time.time()-t0:.2f})')
        with open(os.path.join(meta_path, f'metadata_{year}-{month}.pkl'), 'wb') as f:
            pickle.dump(df, f)

        print(f'> Garbage collection ... ({time.time()-t0:.2f})')
        del embeddings
        del df
        gc.collect()

    print(f'Stopping multiprocess pool ...')
    model.stop_multi_process_pool(pool)

    print(f'\nCOMPLETE. ({time.time()-t0:.2f})\n\n')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=config['embed']['start_year'])
    parser.add_argument('--end-year', type=int, default=config['embed']['end_year'])
    parser.add_argument('--start-month', type=int, default=config['embed']['start_month'])
    parser.add_argument('--end-month', type=int, default=config['embed']['end_month'])
    parser.add_argument('--batch-size', type=int, default=config['embed']['batch_size'])
    parser.add_argument('--show-progress', type=bool, default=config['embed']['show_progress'])
    parser.add_argument('--hf-model', type=str, default=config['embed']['hf_model'])
    parser.add_argument('--embed-path', type=str, default=config['embed']['embed_subpath'])
    parser.add_argument('--meta-path', type=str, default=config['embed']['meta_subpath'])
    
    args = parser.parse_args()

    main(
        data_path=config['general']['data_path'],
        embed_path=os.path.join(config['general']['save_path'], args.embed_path),
        meta_path=os.path.join(config['general']['save_path'], args.meta_path),
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
        hf_model=args.hf_model,
        hf_path=config['general']['hf_path']
    )

