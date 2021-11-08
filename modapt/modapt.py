from os import makedirs
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from modapt.dataset.bow_dataset import build_vocab
from modapt.dataset.common import from_df
from modapt.learning import TRAIN_BATCHSIZE
from modapt.lexicon import train_lexicon_model
from modapt.model.logreg_config.grid_search import load_logreg_model_config_all_archs
from modapt.model.zoo import get_model
from modapt.utils import AUTO_DEVICE, save_json, write_str_list_as_txt


def logreg_train(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "logreg+sn+kb",
    vocab_size: int = 5000,
    use_lemmatize: bool = False,
) -> Tuple[nn.Module, List[str], Dict[str, any]]:
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(data)

    datadef = from_df(df)
    config = load_logreg_model_config_all_archs(datadef.n_classes, datadef.n_sources)[
        arch
    ]
    config["vocab_size"] = vocab_size
    model = get_model(config).to(AUTO_DEVICE)
    use_source_individual_norm = config["use_source_individual_norm"]
    use_lemmatize = config["use_lemmatize"]

    train_samples = datadef.load_splits_func(datadef.domain_names, ["train"])["default"]
    vocab, all_tokens = build_vocab(train_samples, vocab_size, use_lemmatize)

    model, train_metrics = train_lexicon_model(
        model,
        datadef,
        train_samples,
        vocab,
        use_source_individual_norm,
        use_lemmatize,
        "train",
    )

    makedirs(save_model_dir, exist_ok=True)
    write_str_list_as_txt(vocab, join(save_model_dir, "vocab.txt"))
    torch.save(model, join(save_model_dir, "model.pth"))
    save_json(train_metrics, join(save_model_dir, "train_metrics.json"))

    df = model.get_weighted_lexicon(vocab, datadef.label_names)
    df.to_csv(join(save_model_dir, "lexicon.csv"), index=False)

    return model, vocab, train_metrics


def logreg_score(
    data: Union[str, pd.DataFrame],
    model_dir: str,
) -> np.array:
    pass


def logreg_eval(
    data: Union[str, pd.DataFrame],
    model_dir: str,
) -> Tuple[np.array, np.array, Dict[str, any]]:
    pass


def roberta_train(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "roberta+kb",
    n_epoch: int = 6,
    batchsize: int = TRAIN_BATCHSIZE,
) -> Tuple[nn.Module, Dict[str, any]]:
    pass


def roberta_score(
    data: Union[str, pd.DataFrame],
    model_dir: str,
) -> np.array:
    pass


def roberta_eval(
    data: Union[str, pd.DataFrame],
    model_dir: str,
) -> Tuple[np.array, np.array, Dict[str, any]]:
    pass
