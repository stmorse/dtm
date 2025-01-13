"""
Convert .npz to .zarr
"""

import argparse
import configparser
import os
import time

import numpy as np
import dask.array as da

def convert_embeddings(
    embed_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    chunk_size: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'Embedding range           : {years}, {months}')
    print(f'Saving embeds to path     : {embed_path}\n')

    # Load from .npz and convert to .zarr via dask.array
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Loading {year}-{month} ... ({time.time()-t0:.3f})')
        with open(os.path.join(embed_path, f'embeddings_{year}-{month}.npz'), 'rb') as f:
            data = np.load(f)['embeddings']

            # Create Dask array w/ specified chunk size
            print(f'> Converting to zarr ... ({time.time()-t0:.3f})')
            ddata = da.from_array(data, chunks=(chunk_size, -1))

            # Store to Zarr
            ddata.to_zarr(os.path.join(embed_path, f'embeddings_{year}-{month}.zarr'), 
                          overwrite=True)
    
    print(f'Complete. ({time.time()-t0:.3f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, required=True)
    parser.add_argument('--end-month', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, required=True)
    args = parser.parse_args()

    convert_embeddings(
        embed_path=g['embed_path'],
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        chunk_size=args.chunk_size,
    )
