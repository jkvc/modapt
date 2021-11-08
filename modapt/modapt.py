from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch.nn as nn

from modapt.learning import TRAIN_BATCHSIZE


def logreg_train(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "logreg+sn+kb",
    vocab_size: int = 5000,
    use_lemmatize: bool = False,
) -> Tuple[nn.Module, List[str], Dict[str, any]]:
    pass


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
