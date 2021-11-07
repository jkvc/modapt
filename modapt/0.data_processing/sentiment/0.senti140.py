from os.path import join
from random import Random

import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import save_json
from tqdm import tqdm

RNG = Random()
RNG.seed(RANDOM_SEED)

_SUBSAMPLE_SIZE = 10000
_POLARITY_SCORE_TO_LABEL = {
    0: "neg",
    4: "pos",
}

df = pd.read_csv(
    join(
        DATA_DIR,
        "sentiment",
        "raw",
        "senti140",
        "training.1600000.processed.noemoticon.csv",
    ),
)

idxs = RNG.sample(range(len(df)), _SUBSAMPLE_SIZE)

dataset_dict = {}

for idx in tqdm(idxs):
    row = df.iloc[idx]
    tweet_id = row[1]
    text = row[5]

    polarity = row[0]
    if polarity not in _POLARITY_SCORE_TO_LABEL:
        continue
    polarity = _POLARITY_SCORE_TO_LABEL[polarity]

    new_id = f"senti140.{tweet_id}"

    dataset_dict[new_id] = {"id": new_id, "text": text, "polarity": polarity}

save_json(dataset_dict, join(DATA_DIR, "sentiment", "senti140.json"))
