# py <script> <savedir>
import sys
from os import makedirs
from os.path import exists, join
from random import Random

import pandas as pd
from config import RANDOM_SEED
from experiments.datadef.zoo import get_datadef
from modapt.dataset.common import to_df

_DATADEF = get_datadef("sentiment")
_SAVEDIR = sys.argv[1]

_NSAMPLE_PER_DOMAIN = 50

_RNG = Random(RANDOM_SEED)
_TRAIN_DOMAINS = ["airline", "imdb", "senti140", "sst"]
_VALID_DOMAIN = "amazon"


makedirs(_SAVEDIR, exist_ok=True)

train_samples = []
for d in _TRAIN_DOMAINS:
    samples = _DATADEF.load_splits_func([d], ["train"])["train"]
    _RNG.shuffle(samples)
    train_samples.extend(samples[:_NSAMPLE_PER_DOMAIN])
train_df = to_df(train_samples)
train_df.to_csv(join(_SAVEDIR, "train.csv"))


valid_samples = _DATADEF.load_splits_func([_VALID_DOMAIN], ["train"])["train"]
_RNG.shuffle(valid_samples)
valid_samples = valid_samples[:10]
valid_df = to_df(valid_samples)
valid_df.to_csv(join(_SAVEDIR, "valid.csv"))
