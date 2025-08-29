"""
Computes Lambda (RW) and displacement values
Currently written for the single-month resample case
TODO: update for the main case
"""

import argparse
import configparser
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def rw(args):
    year, month = args.year, f'{args.month:02}'

    # load the full C matrix and labels
    label_path = os.path.join(args.sub_path, f'align/align_model_{year}-{month}_labels.npz')
    model_path = os.path.join(args.sub_path, f'models')
    save_path = os.path.join(args.sub_path, f'tab/table_{year}-{month}.csv')

    # load topic group labels
    with open(label_path, 'rb') as f:
        labels_ = np.load(f)['labels']
    
    # load full dimension centroids
    C = []
    for i in range(args.n_periods):
        with open(os.path.join(model_path, f'model_cc_{year}-{month}_{i}.npz'), 'rb') as f:
            cc = np.load(f)['cc']
            C.append(cc.copy())
    C = np.vstack(C)

    print(f'Labels ({labels_.shape}), C ({C.shape}) loaded ...')

    n_labels = len(np.unique(labels_[labels_ >= 0]))

    # labels = []
    # log_ll = []
    # p_value_empirical = []

    res = []

    Cp = PCA(n_components=args.n_components).fit_transform(C)

    for coi in np.unique(labels_):
        print(f'> Label {coi} of {n_labels}...')

        if coi == -1:
            continue

        idx = np.where(labels_ == coi)[0]
        T = len(idx)

        if T < 5:
            continue

        # labels.append(coi)

        # diam and displ
        diam = np.amax([np.linalg.norm(C[i,:] - C[j,:]) for i in idx for j in idx])
        disp = np.linalg.norm(C[idx[-1],:] - C[idx[0],:])

        # rw
        # cmx = C[idx,:]
        cmx = Cp[idx,:]
        steps = cmx[1:,:] - cmx[:-1,:]
        avg_step = np.mean(steps, axis=0)
        cov = np.cov(steps, rowvar=False, ddof=1)
        pinv = np.linalg.pinv(cov)

        obs_lam = 0.5 * (T - 1) * (avg_step.T @ pinv @ avg_step)
        # log_ll.append(obs_lam)

        shuffled_lam = []
        n_permutations = args.n_permutations
        for k in range(n_permutations):
            cmx_c = cmx[:,:]
            np.random.shuffle(cmx_c)
            steps_shuffled = cmx_c[1:, :] - cmx_c[:-1, :]
            avg_step_shuffled = np.mean(steps_shuffled, axis=0)
            lam_shuffled = 0.5 * (T - 1) * (avg_step_shuffled.T @ pinv @ avg_step_shuffled)
            shuffled_lam.append(lam_shuffled)

        # compute p-value
        pve = np.mean(np.array(shuffled_lam) > obs_lam)
        # p_value_empirical.append(pve)

        res.append([coi, T, diam, disp, obs_lam, pve])

    # labels = np.array(labels)
    # log_ll = np.array(log_ll)
    # p_value_empirical = np.array(p_value_empirical)

    # convert to pandas and save
    res = pd.DataFrame(res, columns=['coi', 'size', 'diam', 'disp', 'lam', 'pve'])
    res.to_csv(save_path, index=False)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n_components', type=int, default=45)
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--n_periods', type=int, default=10)
    args = parser.parse_args()

    args.sub_path = os.path.join(g['save_path'], args.sub_path)

    print(f'CPU count              : {os.cpu_count()}')
    print(f'Time period            : {args.year}, {args.month}')
    print(f'Saving results to path : {args.sub_path}\n')

    rw(args)