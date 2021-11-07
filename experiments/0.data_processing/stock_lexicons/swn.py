from os import makedirs
from os.path import join

import pandas as pd
from config import STOCK_LEXICON_DIR
from modapt.utils import read_txt_as_str_list, write_str_list_as_txt
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

swn_raw = read_txt_as_str_list(
    join(STOCK_LEXICON_DIR, "swn", "raw", "SentiWordNet_3.0.0.txt")
)

word2posnegscore = {}

for l in tqdm(swn_raw):
    if l.startswith("#"):
        continue

    _, _, pos_score, neg_score, terms, _ = l.split("\t")
    pos_score, neg_score = float(pos_score), float(neg_score)
    for term in terms.split(" "):
        word, occ = term.split("#")
        if occ != "1":
            continue
        if len(word) <= 2:
            continue
        if not all(c.isalpha() for c in word):
            continue
        if pos_score == 0 and neg_score == 0:
            continue

        word2posnegscore[word] = [neg_score, pos_score]

savedir = join(STOCK_LEXICON_DIR, "swn", "processed")
makedirs(savedir, exist_ok=True)

vocab = list(word2posnegscore.keys())
write_str_list_as_txt(vocab, join(savedir, "vocab.txt"))

df = pd.DataFrame()
df["word"] = vocab
df["neg"] = [word2posnegscore[w][0] for w in vocab]
df["pos"] = [word2posnegscore[w][1] for w in vocab]
df.to_csv(join(savedir, "lexicon.csv"), index=False)
