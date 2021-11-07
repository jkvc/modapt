from os import makedirs
from os.path import join

import pandas as pd
from config import STOCK_LEXICON_DIR
from modapt.utils import read_txt_as_str_list, write_str_list_as_txt
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

liwc_raw = read_txt_as_str_list(
    join(STOCK_LEXICON_DIR, "liwc", "raw", "LIWC2007_English080730.dic")
)

pos_words = [
    w
    for w in read_txt_as_str_list(
        join(STOCK_LEXICON_DIR, "oplex", "raw", "positive-words.txt")
    )
    if len(w) > 2
]
neg_words = [
    w
    for w in read_txt_as_str_list(
        join(STOCK_LEXICON_DIR, "oplex", "raw", "negative-words.txt")
    )
    if len(w) > 2
]

save_dir = join(STOCK_LEXICON_DIR, "oplex", "processed")
makedirs(save_dir, exist_ok=True)

unique_words = list(set(pos_words + neg_words))
write_str_list_as_txt(unique_words, join(save_dir, "vocab.txt"))

df = pd.DataFrame()
df["word"] = unique_words
df["neg"] = [1.0 if lemma in neg_words else 0.0 for lemma in unique_words]
df["pos"] = [1.0 if lemma in pos_words else 0.0 for lemma in unique_words]
df.to_csv(join(save_dir, "lexicon.csv"), index=False)
