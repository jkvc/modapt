from os import makedirs
from os.path import join

import pandas as pd
from config import STOCK_LEXICON_DIR
from modapt.utils import read_txt_as_str_list, write_str_list_as_txt
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

geninq_raw = read_txt_as_str_list(
    join(STOCK_LEXICON_DIR, "geninq", "raw", "dictionary.txt")
)

pos_words, neg_words = [], []

for l in tqdm(geninq_raw[1:]):
    toks = l.split("|")[0].split(" ")

    word = toks[0].lower()
    if "#" in word:
        word = word.split("#")[0]

    if len(word) <= 2:
        continue
    if any(c.isdigit() for c in word):
        continue

    labels = toks[1:]
    if "Pstv" in labels:
        pos_words.append(word)
    if "Ngtv" in labels:
        neg_words.append(word)

save_dir = join(STOCK_LEXICON_DIR, "geninq", "processed")
makedirs(save_dir, exist_ok=True)

unique_words = list(set(pos_words + neg_words))
write_str_list_as_txt(unique_words, join(save_dir, "vocab.txt"))

df = pd.DataFrame()
df["word"] = unique_words
df["neg"] = [1.0 if lemma in neg_words else 0.0 for lemma in unique_words]
df["pos"] = [1.0 if lemma in pos_words else 0.0 for lemma in unique_words]
df.to_csv(join(save_dir, "lexicon.csv"), index=False)
