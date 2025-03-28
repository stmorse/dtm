"""
Streams month chunk at a time into CPU Mem, 
batches into multiple GPU for encoding,
and saves to file in chunks (zarr / Dask)
"""

import argparse
import configparser
import gc
import json
import os
import pickle             # currently not collecting metadata
import time

import bz2                # for .bz2
import lzma               # for .xz
import zstandard as zstd  # for .zst
import zarr

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def open_compressed(file_path):
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(file_path, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.open(file_path, 'rb')
    elif file_path.endswith('.zst'):
        # For .zst, return a stream reader
        f = open(file_path, 'rb')  # Open file in binary mode
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return dctx.stream_reader(f)
        # return dctx.read_to_iter(f)
    else:
        raise ValueError('Unsupported file extension.')

def read_sentences(file_path, chunk_size=10000):
    """
    Read JSON entries from a compressed file, extract the 'body' field,
    and yield chunks of size `chunk_size`.
    """
    buffer = []  # To store 'body' fields in chunks
    byte_buffer = b""  # For handling partial lines in `.zst` files

    with open_compressed(file_path) as f:
        # Iterate over the file
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

                # Add to the chunk buffer
                buffer.append(body)
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []

        # Handle any remaining partial JSON line
        if byte_buffer:
            entry = json.loads(byte_buffer.decode("utf-8"))
            if 'body' in entry and entry['author'] != '[deleted]':
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]
                buffer.append(body)

        # Yield any leftovers in the chunk buffer
        if buffer:
            yield buffer


def main(
    data_path: str,
    embed_path: str,
    meta_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    chunk_size: int,
    show_progress: bool,
    hf_model: str,
    hf_path: str
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    os.environ['HF_HOME'] = hf_path

    print(f'GPU enabled?              : {torch.cuda.is_available()}')
    print(f'Embedding range           : {years}, {months}')
    print(f'Saving embeddings to path : {embed_path}')
    # print(f'Saving metadata to path   : {meta_path}\n')
    print(f'Loading model ... ({time.time()-t0:.2f})')

    model = SentenceTransformer(hf_model,
                                device='cuda',
                                model_kwargs={'torch_dtype': 'float16'}) 
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f'Model embedding dimension: {embedding_dim}')

    print(f'Starting multiprocessing pool ... ({time.time()-t0:.2f})')  
    pool = model.start_multi_process_pool()
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Processing {year}-{month} ... ({time.time()-t0:.2f})')

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

        # Open the compressed file
        print(f'> Loading comments (Batch size: {chunk_size}) ... ({time.time()-t0:.3f})')

        # IF ZARR

        # initialize zarr
        zarr_store = zarr.open(
            os.path.join(embed_path, f'embeddings_{year}-{month}.zarr'),
            mode='w',
            shape=(0, embedding_dim),
            chunks=(chunk_size, embedding_dim),
            dtype=np.float32
        )

        # Stream sentences, embed, and append to Zarr
        total_count, chunk_count = 0, 0
        for chunk in read_sentences(file_path, chunk_size=chunk_size):
            # Encode sentences in this chunk (last chunk will be leftovers)
            # embeddings = model.encode(chunk_of_sents, convert_to_numpy=True)
            embeddings = model.encode_multi_process(
                chunk, 
                pool, 
                # convert_to_numpy=True,    # this was in example but doesn't work
                show_progress_bar=show_progress
            )

            zarr_store.append(embeddings)

            total_count = zarr_store.shape[0]
            print(f"> Processed chunk {chunk_count}, total {total_count} \
                  ... ({time.time()-t0:.2f})")
            chunk_count += 1
       
        # IF NPZ
        # initialize an array to store embeddings
        # embeddings = []

        # will hold metadata
        # df = []
        
        # j, k = 0, 0   # line count, batch count
        # with bz2.BZ2File(file_path, 'rb') as f:
        #     batch = []
        #     for line in f:
        #         entry = json.loads(line)
        #         j += 1

        #         if 'body' not in entry or entry['author'] == '[deleted]':
        #             continue
                
        #         # quick and dirty to keep entries closer to SBERT token limit (256)
        #         body = entry['body']
        #         if len(body) > 2000:
        #             body = body[:2000]

        #         batch.append(body)

        #         # continue building metadata
        #         df.append([k, entry['author'], entry['id'], entry['created_utc']])
                        
        #         # when the batch is full, process it
        #         if len(batch) == batch_size:
        #             print(f'> Processing batch {k} ({len(batch)})... ({time.time()-t0:.2f})')

        #             # encode the batch of sentences on the GPU
        #             batch_embeddings = model.encode_multi_process(
        #                 batch, pool, show_progress_bar=show_progress)
        #             embeddings.append(batch_embeddings)
                    
        #             batch = []  # Clear batch after processing
        #             k += 1
            
        #     # Process any remaining sentences in the final batch
        #     if len(batch) > 0:
        #         print(f'> Leftovers ({len(batch)})... ({time.time()-t0:.2f})')

        #         batch_embeddings = model.encode_multi_process(
        #                 batch, pool, show_progress_bar=show_progress)
                
        #         embeddings.append(batch_embeddings)
        #         batch = []
        #         k += 1

        # Convert list of arrays into a single array
        # embeddings = np.vstack(embeddings)

        # Save the embeddings to disk
        # print(f'> Total lines: {j}  Total embeddings: {len(embeddings)}  Total batches: {k}')
        # print(f'> Saving to disk ... ({time.time()-t0:.2f})')
        # with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'wb') as f:
        #     np.savez_compressed(f, embeddings=embeddings, allow_pickle=False)

        # save metadata to disk
        # print(f'> Saving metadata to disk ... ({time.time()-t0:.2f})')
        # with open(os.path.join(meta_path, f'metadata_{year}-{month}.pkl'), 'wb') as f:
        #     pickle.dump(df, f)

        print(f'> Garbage collection ... ({time.time()-t0:.2f})')
        del embeddings
        # del df
        gc.collect()

    print(f'Stopping multiprocess pool ...')
    model.stop_multi_process_pool(pool)

    print(f'\nCOMPLETE. ({time.time()-t0:.2f})\n\n')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']
    c = config['embed']

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed-path', type=str, default=g['embed_path'])
    parser.add_argument('--meta-path', type=str, default=g['meta_path'])
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--start-month', type=int)
    parser.add_argument('--end-month', type=int)
    parser.add_argument('--chunk-size', type=int, default=c['chunk_size'])
    parser.add_argument('--show-progress', type=bool, default=False)
    parser.add_argument('--hf-model', type=str, default=c['hf_model'])
    
    args = parser.parse_args()

    main(
        data_path=g['data_path'],
        embed_path=args.embed_path,
        meta_path=args.meta_path,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        chunk_size=args.chunk_size,
        show_progress=args.show_progress,
        hf_model=args.hf_model,
        hf_path=g['hf_path']
    )

