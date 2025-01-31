import argparse
import configparser
import os
import pickle
import time

import numpy as np
import umap
from sklearn.cluster import HDBSCAN


def rw(
    align_path: str,
):
    with open(os.path.join(align_path, 'align_model_HDBSCAN.pkl'), 'rb') as f:
        align_model = pickle.load(f)