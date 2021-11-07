import base64
import hashlib
from os.path import join
from random import Random

import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import save_json
from tqdm import tqdm

RNG = Random()
RNG.seed(RANDOM_SEED)

_SUBSAMPLE_SIZE = 10000
_POLARITY_TO_LABEL = {"positive": "pos", "negative": "neg"}

df = pd.read_csv(
    join(DATA_DIR, "sentiment", "raw", "imdb", "IMDB Dataset.csv"),
)

idxs = RNG.sample(range(len(df)), _SUBSAMPLE_SIZE)

dataset_dict = {}

for idx in tqdm(idxs):
    row = df.iloc[idx]
    text = row[0]
    polarity = row[1]
    polarity = _POLARITY_TO_LABEL[polarity]

    hasher = hashlib.sha1(text.encode())
    review_id = base64.urlsafe_b64encode(hasher.digest()[:6]).decode()

    print(review_id, polarity, text)

    new_id = f"imdb.{review_id}"

    dataset_dict[new_id] = {"id": new_id, "text": text, "polarity": polarity}

save_json(dataset_dict, join(DATA_DIR, "sentiment", "imdb.json"))
