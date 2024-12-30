"""
Loads clusters labels and sentences
Trains tfidf on every cluster
Saves to file
"""

import argparse
import configparser
import gc
import time
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def train_tfidf(
    data_path: str,
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


if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    c = config['tfidf']

    parser = argparse.ArgumentParser()
    parser.add_argument('--label-path', type=str, default=c['label_subpath'])
    parser.add_argument('--tfidf-path', type=str, default=c['tfidf_subpath'])
    parser.add_argument('--start-year', type=int, default=c['start_year'])
    parser.add_argument('--end-year', type=int, default=c['end_year'])
    parser.add_argument('--start-month', type=int, default=c['start_month'])
    parser.add_argument('--end-month', type=int, default=c['end_month'])
    parser.add_argument('--chunk-size', type=int, default=c['chunk_size'])
    
    args = parser.parse_args()

    train_tfidf(
        data_path=config['general']['data_path'],
        label_path=os.path.join(config['general']['save_path'], args.label_path),
        tfidf_path=os.path.join(config['general']['save_path'], args.tfidf_path),
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        chunk_size=args.chunk_size,
    )
