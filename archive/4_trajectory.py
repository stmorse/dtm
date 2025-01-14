"""
Load 2 months cluster tfidf
Compare and build bipartite graph based on similarity score
Load next two months, etc
Output for each month: 
{
    'year': int, 
    'month': int,
    'cluster_map': {
        cluster # current month: cluster # next month,
    }
    'similarity': ndarray
}
"""

import configparser
import os
import pickle
import time

import numpy as np


def jaccard_score(set1, set2):
    # set intersection is &, union is |
    # TODO: implement fuzzy matching
    set1, set2 = set(set1), set(set2)
    return len(set1 & set2) / len(set1 | set2)

def compute_trajectory(
    tfidf_path: str,
    maps_path: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
):
    t0 = time.time()
    years = [str(y) for y in range(start_year, end_year+1)]
    months = [f'{m:02}' for m in range(start_month, end_month+1)]

    print(f'CPU count                 : {os.cpu_count()}')
    print(f'Trajectory range          : {years}, {months}')
    print(f'Using tf-idf from         : {tfidf_path}')
    print(f'Saving similarity maps to : {maps_path}\n')

    yrmos = [(yr, mo) for yr in years for mo in months]
    
    # get first month
    cur_yr, cur_mo = yrmos[0]
    with open(os.path.join(tfidf_path, f'tfidf_{cur_yr}-{cur_mo}.pkl'), 'rb') as f:
        current_tfidf = pickle.load(f)['tfidf']
    
    # get num clusters (right now is constant but later may change)
    nc1 = len(current_tfidf.keys())

    for ymx in range(1, len(yrmos)):
        print(f'Processing {cur_yr}-{cur_mo} ... ({time.time()-t0:.3f})')

        next_yr, next_mo = yrmos[ymx]
        with open(os.path.join(tfidf_path, f'tfidf_{next_yr}-{next_mo}.pkl'), 'rb') as f:
            next_tfidf = pickle.load(f)['tfidf']
        nc2 = len(next_tfidf.keys())

        # create pairwise similarity matrix
        similarity = np.zeros((nc1, nc2))
        for i in range(nc1):
            for j in range(nc2):
                similarity[i,j] = jaccard_score(
                    current_tfidf[i]['keywords'],
                    next_tfidf[j]['keywords']
                )

        # create map based on top match
        cluster_map = {}
        for i in range(nc1):
            cluster_map[i] = np.argmax(similarity[i,:])

        # save to disk
        output = {
            'year': cur_yr,
            'month': cur_mo,
            'cluster_map': cluster_map,
            'similarity': similarity
        }
        with open(os.path.join(maps_path, f'maps_{cur_yr}-{cur_mo}.pkl'), 'wb') as f:
            pickle.dump(output, f)

        cur_yr, cur_mo = next_yr, next_mo
        nc1 = nc2
        current_tfidf = next_tfidf

    print(f'Complete. ({time.time()-t0:.3f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    c = config['trajectory']

    compute_trajectory(
        tfidf_path=os.path.join(config['general']['save_path'], c['tfidf_subpath']),
        maps_path=os.path.join(config['general']['save_path'], c['maps_subpath']),
        start_year=int(c['start_year']),
        end_year=int(c['end_year']),
        start_month=int(c['start_month']),
        end_month=int(c['end_month']),
    )