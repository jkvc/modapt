from os.path import join
from random import Random

import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import save_json
from tqdm import tqdm

RNG = Random()
RNG.seed(RANDOM_SEED)

_SUBSAMPLE_SIZE = 10000
_POLARITY_TO_LABEL = {
    "positive": "pos",
    # "neutral": "pos",  # call neutral positive for balance
    "negative": "neg",
}

df = pd.read_csv(
    join(DATA_DIR, "sentiment", "raw", "airline", "Tweets.csv"),
)

idxs = RNG.sample(range(len(df)), _SUBSAMPLE_SIZE)

dataset_dict = {}

for idx in tqdm(idxs):
    row = df.iloc[idx]
    text = row[10]
    polarity = row[1]
    if polarity not in _POLARITY_TO_LABEL:
        continue
    polarity = _POLARITY_TO_LABEL[polarity]
    tweet_id = row[0]

    print(tweet_id, polarity, text)

    new_id = f"airline.{tweet_id}"

    dataset_dict[new_id] = {"id": new_id, "text": text, "polarity": polarity}

save_json(dataset_dict, join(DATA_DIR, "sentiment", "airline.json"))
