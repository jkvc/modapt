import glob
import gzip
import json
from os import makedirs
from os.path import basename, join
from posixpath import splitext
from random import Random, randint, shuffle

import numpy as np
import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import ParallelHandler, load_json, save_json
from tqdm import tqdm

_TRAIN_PROP, _VALID_PROP, _TEST_PROP = [0.8, 0.1, 0.1]


_SRC_DATA_DIR = join(DATA_DIR, "sentiment")
_SPLITS_DIR = join(_SRC_DATA_DIR, "splits")


RNG = Random()
RNG.seed(RANDOM_SEED)


makedirs(_SPLITS_DIR, exist_ok=True)

raw_data_paths = sorted(glob.glob(join(_SRC_DATA_DIR, "*.json")))
for p in raw_data_paths:
    name = splitext(basename(p))[0]
    samples = load_json(p)
    ids = list(samples.keys())
    RNG.shuffle(ids)

    nsample = len(ids)
    n_train = int(nsample * _TRAIN_PROP)
    n_valid = int(nsample * _VALID_PROP)
    n_test = nsample - n_train - n_valid
    save_json(ids[:n_train], join(_SPLITS_DIR, f"{name}.train.json"))
    save_json(ids[n_train : n_train + n_valid], join(_SPLITS_DIR, f"{name}.valid.json"))
    save_json(ids[-n_test:], join(_SPLITS_DIR, f"{name}.test.json"))
