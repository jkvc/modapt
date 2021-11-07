# use LIWC2007_English080730.dic, lemmatize
# keep words in vocab containing label 126 posemo and 127 negemo


from os import makedirs
from os.path import join

import pandas as pd
from config import STOCK_LEXICON_DIR
from modapt.utils import read_txt_as_str_list, write_str_list_as_txt
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

if __name__ == "__main__":
    liwc_raw = read_txt_as_str_list(
        join(STOCK_LEXICON_DIR, "liwc", "raw", "LIWC2007_English080730.dic")
    )
    lemmatizer = WordNetLemmatizer()

    pos_lemmas, neg_lemmas = [], []
    for l in tqdm(liwc_raw):
        if l == "%":
            continue
        tokens = l.split("\t")
        if tokens[0].isnumeric():
            continue

        token = lemmatizer.lemmatize(tokens[0].replace("*", ""))

        labels_str = " ".join(tokens[1:])
        if "126" in labels_str:
            pos_lemmas.append(token)
        if "127" in labels_str:
            neg_lemmas.append(token)

    save_dir = join(STOCK_LEXICON_DIR, "liwc", "processed")
    makedirs(save_dir, exist_ok=True)

    unique_lemmas = list(set(pos_lemmas + neg_lemmas))
    write_str_list_as_txt(unique_lemmas, join(save_dir, "vocab.txt"))

    df = pd.DataFrame()
    df["word"] = unique_lemmas
    df["neg"] = [1.0 if lemma in neg_lemmas else 0.0 for lemma in unique_lemmas]
    df["pos"] = [1.0 if lemma in pos_lemmas else 0.0 for lemma in unique_lemmas]
    df.to_csv(join(save_dir, "lexicon.csv"), index=False)
