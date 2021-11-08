from os.path import dirname, join, realpath

import pandas as pd
from modapt.modapt import logreg_train

_THIS_DIR = dirname(realpath(__file__))

print(">> training logreg model")

model, vocab, metrics = logreg_train(
    data=join(_THIS_DIR, "demo_data", "train.csv"),
    save_model_dir=join(_THIS_DIR, "models"),
    vocab_size=100,
    use_lemmatize=True,
)
