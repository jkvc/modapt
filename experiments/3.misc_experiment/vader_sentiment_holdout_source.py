# Usage: python <script_name>


import sys
from os import makedirs
from os.path import join
from posixpath import dirname
from random import Random

from config import LEXICON_DIR, RANDOM_SEED
from experiments.datadef.zoo import get_datadef
from modapt.dataset.bow_dataset import get_tokens
from modapt.eval import reduce_and_save_metrics
from modapt.utils import save_json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

RNG = Random()
RNG.seed(RANDOM_SEED)

_DATASET_NAME = "sentiment"
_DATADEF = get_datadef(_DATASET_NAME)
_SAVE_DIR = join(LEXICON_DIR, _DATASET_NAME, "holdout_source", "vader")


_ANALYZER = SentimentIntensityAnalyzer()

for holdout_source in _DATADEF.domain_names:
    print(">>", holdout_source)
    logdir = join(_SAVE_DIR, holdout_source)
    makedirs(logdir, exist_ok=True)

    # valid using holdout issue all samples
    valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]

    num_correct = 0
    for s in tqdm(valid_samples):
        text = " ".join(get_tokens(s.text))
        score = _ANALYZER.polarity_scores(text)["compound"]
        is_correct = (
            (score > 0 and s.y_idx == 1)
            or (score < 0 and s.y_idx == 0)
            or (score == 0 and RNG.uniform(0, 1) > 0.5)  # random break tie
        )
        if is_correct:
            num_correct += 1
    acc = num_correct / len(valid_samples)

    metrics = {
        "valid_f1": acc,
        "valid_precision": acc,
        "valid_recall": acc,
    }
    save_json(metrics, join(logdir, "leaf_metrics.json"))

reduce_and_save_metrics(dirname(_SAVE_DIR))
