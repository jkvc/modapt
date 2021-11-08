from os import makedirs
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from modapt.dataset.bow_dataset import build_bow_full_batch, build_vocab, get_all_tokens
from modapt.dataset.common import (
    calculate_labelprops,
    from_labeled_df,
    from_unlabeled_df,
)
from modapt.dataset.dataset_def import DatasetDefinition
from modapt.learning import TRAIN_BATCHSIZE
from modapt.lexicon import eval_lexicon_model, train_lexicon_model
from modapt.model.logreg_config.grid_search import load_logreg_model_config_all_archs
from modapt.model.zoo import get_model
from modapt.utils import (
    AUTO_DEVICE,
    load_json,
    read_txt_as_str_list,
    save_json,
    write_str_list_as_txt,
)


def _get_data_df(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(data)
    return df


def logreg_train(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "logreg+sn+kb",
    vocab_size: int = 5000,
) -> Tuple[nn.Module, List[str], Dict[str, float]]:
    df = _get_data_df(data)
    datadef = from_labeled_df(df)

    config = load_logreg_model_config_all_archs(datadef.n_classes, datadef.n_sources)[
        arch
    ]
    config["vocab_size"] = vocab_size
    model = get_model(config).to(AUTO_DEVICE)
    use_source_individual_norm = config["use_source_individual_norm"]
    use_lemmatize = config["use_lemmatize"]

    train_samples = datadef.load_splits_func(datadef.domain_names, ["train"])["default"]
    vocab, all_tokens = build_vocab(train_samples, vocab_size, use_lemmatize)

    print(f">> training logreg model with {len(train_samples)} samples")

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

    save_json(config, join(save_model_dir, "config.json"))

    return model, vocab, train_metrics


def logreg_predict_eval(
    data_labeled: Union[str, pd.DataFrame],
    data_unlabeled: Union[str, pd.DataFrame],
    model_dir: str,
) -> Tuple[np.array, float]:
    model = torch.load(join(model_dir, "model.pth"))
    vocab = read_txt_as_str_list(join(model_dir, "vocab.txt"))
    config = load_json(join(model_dir, "config.json"))

    df_labeled = _get_data_df(data_labeled)
    datadef_labeled = from_labeled_df(df_labeled)
    df_unlabeled = _get_data_df(data_unlabeled)
    datadef_unlabeled = from_unlabeled_df(
        df_unlabeled, nclasses=datadef_labeled.n_classes
    )

    # labeled samples
    labeled_samples = datadef_labeled.load_splits_func(
        datadef_labeled.domain_names, [""]
    )["default"]

    # unlabeled samples use labelprops calculated from all labeled samples
    estimated_labelprops = {
        "estimated": calculate_labelprops(
            labeled_samples,
            datadef_labeled.n_classes,
            datadef_labeled.domain_names,
        )
    }
    datadef_unlabeled.load_labelprops_func = lambda _split: estimated_labelprops[_split]
    unlabeled_samples = datadef_unlabeled.load_splits_func(
        datadef_unlabeled.domain_names, [""]
    )["default"]

    # do prediction
    print(f">> running prediction on {len(unlabeled_samples)} unlabeled samples")
    batch = build_bow_full_batch(
        samples=unlabeled_samples,
        datadef=datadef_unlabeled,
        all_tokens=get_all_tokens(
            unlabeled_samples, use_lemmatize=config["use_lemmatize"]
        ),
        vocab=vocab,
        use_source_individual_norm=config["use_source_individual_norm"],
        labelprop_split="estimated",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    logits = outputs["logits"]
    scores = F.softmax(logits, dim=-1).detach().cpu().numpy()

    # estimate acc
    print(f">> estimating accuracy using {len(labeled_samples)} labeled samples")
    halfsize = len(labeled_samples) // 2
    firsthalf, secondhalf = labeled_samples[:halfsize], labeled_samples[halfsize:]
    accs = []
    for labelprop_est_samples, valid_samples in [
        (firsthalf, secondhalf),
        (secondhalf, firsthalf),
    ]:
        estimated_labelprops = {
            "estimated": calculate_labelprops(
                labelprop_est_samples,
                datadef_labeled.n_classes,
                datadef_labeled.domain_names,
            )
        }
        tmp_datadef = DatasetDefinition(
            domain_names=datadef_labeled.domain_names,
            label_names=datadef_labeled.label_names,
            load_splits_func=datadef_labeled.load_splits_func,
            load_labelprops_func=lambda _split: estimated_labelprops[_split],
        )
        metrics = eval_lexicon_model(
            model,
            tmp_datadef,
            valid_samples,
            vocab,
            use_source_individual_norm=config["use_source_individual_norm"],
            labelprop_split="estimated",  # match _load_labelprops_func()
            use_lemmatize=config["use_lemmatize"],
        )
        accs.append(metrics["valid_f1"])
    est_acc = sum(accs) / len(accs)

    return scores, est_acc


def roberta_train(
    data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "roberta+kb",
    n_epoch: int = 6,
    batchsize: int = TRAIN_BATCHSIZE,
) -> Tuple[nn.Module, Dict[str, any]]:
    pass


def roberta_eval(
    data: Union[str, pd.DataFrame],
    model_dir: str,
) -> Tuple[np.array, np.array, Dict[str, any]]:
    pass
