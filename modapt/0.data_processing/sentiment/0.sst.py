from os.path import join
from random import Random

import pandas as pd
from config import DATA_DIR, RANDOM_SEED
from modapt.utils import read_txt_as_str_list, save_json
from tqdm import tqdm

id2sentiscore = {}
lines = read_txt_as_str_list(
    join(DATA_DIR, "sentiment", "raw", "sst", "sentiment_labels.txt")
)
for l in lines[1:]:
    phrase_id, sentiscore = l.split("|")
    id2sentiscore[int(phrase_id)] = float(sentiscore)


def sentiscore2polarity(sentiscore):
    if sentiscore < 0.3:
        return "neg"
    elif sentiscore > 0.7:
        return "pos"
    else:
        return None


dataset_dict = {}

df = pd.read_csv(
    join(DATA_DIR, "sentiment", "raw", "sst", "datasetSentences.txt"), sep="\t"
)
for _, row in df.iterrows():
    phrase_id = row[0]
    text = row[1]
    sentiscore = id2sentiscore[phrase_id]
    polarity = sentiscore2polarity(sentiscore)
    if polarity is None:
        continue

    new_id = f"sst.{phrase_id}"
    dataset_dict[new_id] = {"id": new_id, "text": text, "polarity": polarity}

save_json(dataset_dict, join(DATA_DIR, "sentiment", "sst.json"))
