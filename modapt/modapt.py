from typing import Union

import pandas as pd

from modapt.learning import TRAIN_BATCHSIZE


def train_logreg(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "logreg+sn+kb",
    vocab_size: int = 5000,
    use_lemmatize: bool = False,
):
    pass


def eval_logreg(
    data: Union[str, pd.DataFrame],
    model_dir: str,
):
    pass


def train_roberta(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "roberta+kb",
    n_epoch: int = 6,
    batchsize: int = TRAIN_BATCHSIZE,
):
    pass


def eval_roberta(
    data: Union[str, pd.DataFrame],
    model_dir: str,
):
    pass
