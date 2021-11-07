import gzip
import json
import sys
from os.path import join
from random import Random

import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import read_txt_as_str_list, save_json
from tqdm import tqdm

_PATH = sys.argv[1]

RNG = Random()
RNG.seed(RANDOM_SEED)

_TOTAL_SAMPLES = 1600000
_SUBSAMPLE_SIZE = 10000
_RATING_TO_LABEL = {
    1: "neg",
    2: "neg",
    # 3: "neg",
    4: "pos",
    5: "pos",
}

idxs = set(RNG.sample(range(_TOTAL_SAMPLES), _SUBSAMPLE_SIZE))
samples = []

g = gzip.open(_PATH, "r")
for i, l in enumerate(tqdm(g)):
    if i > _TOTAL_SAMPLES:
        break
    if i in idxs:
        samples.append(eval(l))

dataset_dict = {}

for sample in tqdm(samples):

    text = sample["reviewText"]
    rating = int(sample["overall"])
    if rating not in _RATING_TO_LABEL:
        continue

    polarity = _RATING_TO_LABEL[rating]
    new_id = f"amazon.{sample['asin']}-{sample['reviewerID']}"

    dataset_dict[new_id] = {"id": new_id, "text": text, "polarity": polarity}

save_json(dataset_dict, join(DATA_DIR, "sentiment", "amazon.json"))
