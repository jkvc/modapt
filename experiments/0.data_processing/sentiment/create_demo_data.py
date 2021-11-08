# py <script> <savedir>
import sys
from os import makedirs
from os.path import join
from random import Random

import pandas as pd
from config import RANDOM_SEED
from experiments.datadef.zoo import get_datadef
from modapt.dataset.common import to_df

_DATADEF = get_datadef("sentiment")
_SAVEDIR = sys.argv[1]

_N_TRAIN_SAMPLE_PER_DOMAIN = 100
_N_VALID_LABELED = 20
_N_VALID_UNLABELED = 20

_RNG = Random(RANDOM_SEED)
_TRAIN_DOMAINS = ["airline", "imdb", "senti140", "sst"]
_VALID_DOMAIN = "amazon"


makedirs(_SAVEDIR, exist_ok=True)

train_samples = []
for d in _TRAIN_DOMAINS:
    samples = _DATADEF.load_splits_func([d], ["train"])["train"]
    _RNG.shuffle(samples)
    train_samples.extend(samples[:_N_TRAIN_SAMPLE_PER_DOMAIN])
train_df = to_df(train_samples)
train_df.to_csv(join(_SAVEDIR, "train.csv"))


valid_samples = _DATADEF.load_splits_func([_VALID_DOMAIN], ["train"])["train"]

_RNG.shuffle(valid_samples)
valid_labeled_samples = valid_samples[:_N_VALID_LABELED]
valid_labeled_df = to_df(valid_labeled_samples)
valid_labeled_df.to_csv(join(_SAVEDIR, "valid_labeled.csv"))

_RNG.shuffle(valid_samples)
valid_unlabeled_samples = valid_samples[:_N_VALID_LABELED]
valid_unlabeled_df = to_df(valid_unlabeled_samples)
valid_unlabeled_df = valid_unlabeled_df.drop("y_idx", 1)
valid_unlabeled_df.to_csv(join(_SAVEDIR, "valid_unlabeled.csv"))
