# Usage: python <script_name> <dataset_name> <stock_lexicon_name> <arch>


import sys
from os import makedirs
from os.path import basename, join, realpath
from posixpath import dirname

import pandas as pd
import torch
from config import LEXICON_DIR, STOCK_LEXICON_DIR
from experiments.datadef.zoo import get_datadef
from modapt.eval import reduce_and_save_metrics
from modapt.lexicon import eval_lexicon_model, train_lexicon_model
from modapt.model import get_model
from modapt.model.logreg_config.base import (
    load_stock_lexicon_logreg_model_config,
)
from modapt.utils import (
    AUTO_DEVICE,
    read_txt_as_str_list,
    save_json,
    write_str_list_as_txt,
)

_DATASET_NAME = sys.argv[1]
_STOCK_LEXICON_NAME = sys.argv[2]
_ARCH = sys.argv[3]
assert "+ab" in _ARCH

_DATADEF = get_datadef(_DATASET_NAME)

_SAVE_DIR = join(
    LEXICON_DIR, _DATASET_NAME, "holdout_source", f"{_STOCK_LEXICON_NAME}@{_ARCH}"
)

_USE_ADAPT_NUM_SAMPLES = 250
_N_TRIALS = 5

# load stock lexicon
stock_lexicon_dir = join(STOCK_LEXICON_DIR, _STOCK_LEXICON_NAME, "processed")
vocab = read_txt_as_str_list(join(stock_lexicon_dir, "vocab.txt"))
vocab_size = len(vocab)
lexicon_df = pd.read_csv(join(stock_lexicon_dir, "lexicon.csv"))
print(lexicon_df)

for holdout_source in _DATADEF.domain_names:
    print(">>", holdout_source)
    logdir = join(_SAVE_DIR, holdout_source)
    makedirs(logdir, exist_ok=True)

    # valid using holdout issue all samples
    valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]

    for ti in range(_N_TRIALS):
        trial_logdir = join(logdir, f"trial_{ti}")
        makedirs(trial_logdir, exist_ok=True)

        # build model
        config = load_stock_lexicon_logreg_model_config(
            lexicon_name=_STOCK_LEXICON_NAME,
            arch=_ARCH,
            n_classes=_DATADEF.n_classes,
            n_sources=_DATADEF.n_sources,
            vocab_size=len(vocab),
        )
        model = get_model(config).to(AUTO_DEVICE)
        model.set_weight_from_lexicon(lexicon_df, _DATADEF.label_names)

        use_source_individual_norm = config["use_source_individual_norm"]
        use_lemmatize = config["use_lemmatize"]

        metrics = {}

        # train with 250
        train_samples = valid_samples[
            ti * _USE_ADAPT_NUM_SAMPLES : (ti + 1) * _USE_ADAPT_NUM_SAMPLES
        ]
        model, train_metrics = train_lexicon_model(
            model,
            _DATADEF,
            train_samples,
            vocab,
            use_source_individual_norm,
            use_lemmatize,
            labelprop_split="train",
        )
        metrics.update(train_metrics)

        # run validation set
        valid_metrics = eval_lexicon_model(
            model=model,
            datadef=_DATADEF,
            valid_samples=valid_samples,
            vocab=vocab,
            use_source_individual_norm=use_source_individual_norm,
            use_lemmatize=use_lemmatize,
            labelprop_split="train",
        )
        metrics.update(valid_metrics)
        save_json(metrics, join(trial_logdir, "leaf_metrics.json"))
        write_str_list_as_txt(vocab, join(trial_logdir, "vocab.txt"))
        torch.save(model, join(trial_logdir, "model.pth"))


save_json(config, join(_SAVE_DIR, "config.json"))

reduce_and_save_metrics(dirname(_SAVE_DIR))
reduce_and_save_metrics(dirname(_SAVE_DIR), "leaf_test.json", "mean_test.json")
