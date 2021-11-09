from os import makedirs
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from modapt.dataset.bow_dataset import build_bow_full_batch, build_vocab, get_all_tokens
from modapt.dataset.common import (
    calculate_labelprops,
    from_labeled_df,
    from_unlabeled_df,
)
from modapt.dataset.dataset_def import DatasetDefinition
from modapt.dataset.roberta_dataset import RobertaDataset
from modapt.learning import (
    N_DATALOADER_WORKER,
    TRAIN_BATCHSIZE,
    VALID_BATCHSIZE,
    do_valid,
    train,
)
from modapt.lexicon import eval_lexicon_model, train_lexicon_model
from modapt.model.logreg_config.grid_search import load_logreg_model_config_all_archs
from modapt.model.roberta_config.base import load_roberta_model_config
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
    """train a logistic regression model from bag-of-word features

    Args:
        data (Union[str, pd.DataFrame]): dataframe or path to dataframe of training data,
            see `dataset.common.from_labeled_df` for format
        save_model_dir (str): dirname to save model
        arch (str, optional): model architecture, see `model.logreg_config.grid_search`. Defaults to "logreg+sn+kb".
        vocab_size (int, optional): vocab size. Defaults to 5000.

    Returns:
        Tuple[nn.Module, List[str], Dict[str, float]]: model, vocab list, metrics
    """
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
    """predict using a trained logreg model, and estimate its performance

    Args:
        data_labeled (Union[str, pd.DataFrame]): dataframe or path to csv of labeled data for performance estimation,
            see `dataset.common.from_labeled_df`
        data_unlabeled (Union[str, pd.DataFrame]): dataframe or path to csv of unlabeled data,
            see `dataset.common.from_unlabeled_df`
        model_dir (str): path to directory of saved model

    Returns:
        Tuple[np.array, float]: scores, estimated accuracy
    """
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
    train_data: Union[str, pd.DataFrame],
    valid_data: Union[str, pd.DataFrame],
    save_model_dir: str,
    arch: str = "roberta+kb",
    max_epochs: int = 6,
    batchsize: int = TRAIN_BATCHSIZE,
) -> Tuple[nn.Module, Dict[str, any]]:
    """train a roberta model from hugginface public checkpoint

    Args:
        train_data (Union[str, pd.DataFrame]): dataframe or path to dataframe of training data,
            see `dataset.common.from_labeled_df` for format
        valid_data (Union[str, pd.DataFrame]): dataframe or path to dataframe of validation data, used for early-stopping,
            see `dataset.common.from_labeled_df` for format
        save_model_dir (str): dirname to save model
        arch (str, optional): model architecture, see `model.roberta_config`. Defaults to "roberta+kb".
        max_epochs (int, optional): train at most this many epoch if no early stop. Defaults to 6.
        batchsize (int, optional): batchsize. Defaults to TRAIN_BATCHSIZE.

    Returns:
        Tuple[nn.Module, Dict[str, any]]: model, metrics
    """

    train_df = _get_data_df(train_data)
    train_datadef = from_labeled_df(train_df)
    valid_df = _get_data_df(valid_data)
    valid_datadef = from_labeled_df(valid_df)

    config = load_roberta_model_config(
        arch, train_datadef.n_classes, train_datadef.n_sources
    )

    train_samples = train_datadef.load_splits_func(
        train_datadef.domain_names, ["train"]
    )["default"]
    train_dataset = RobertaDataset(
        train_samples,
        n_classes=train_datadef.n_classes,
        domain_names=train_datadef.domain_names,
        source2labelprops=train_datadef.load_labelprops_func("train"),
    )
    valid_samples = valid_datadef.load_splits_func(
        valid_datadef.domain_names, ["valid"]
    )["default"]
    valid_dataset = RobertaDataset(
        valid_samples,
        n_classes=valid_datadef.n_classes,
        domain_names=valid_datadef.domain_names,
        source2labelprops=valid_datadef.load_labelprops_func("valid"),
    )

    makedirs(save_model_dir, exist_ok=True)
    model = get_model(config)

    print(f">> training roberta model with {len(train_samples)} train samples")
    print(f">>         early stopping with {len(valid_samples)} valid samples")
    metrics = train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        logdir=save_model_dir,
        max_epochs=max_epochs,
        num_early_stop_non_improve_epoch=max_epochs,
        batchsize=batchsize,
    )

    save_json(config, join(save_model_dir, "config.json"))
    return model, metrics


def roberta_predict_eval(
    data_labeled: Union[str, pd.DataFrame],
    data_unlabeled: Union[str, pd.DataFrame],
    model_dir: str,
) -> Tuple[np.array, float]:
    """predict using a trained roberta model, and estimate its performance

    Args:
        data_labeled (Union[str, pd.DataFrame]): dataframe or path to csv of labeled data for performance estimation,
            see `dataset.common.from_labeled_df`
        data_unlabeled (Union[str, pd.DataFrame]): dataframe or path to csv of unlabeled data,
            see `dataset.common.from_unlabeled_df`
        model_dir (str): path to directory of saved model

    Returns:
        Tuple[np.array, float]: scores, estimated accuracy
    """
    model = torch.load(join(model_dir, "checkpoint.pth"))

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
    full_unlabeled_dataset = RobertaDataset(
        samples=unlabeled_samples,
        n_classes=datadef_labeled.n_classes,
        domain_names=datadef_labeled.domain_names,
        source2labelprops=datadef_labeled.load_labelprops_func("estimated"),
    )
    full_unlabeled_loader = DataLoader(
        full_unlabeled_dataset,
        batch_size=VALID_BATCHSIZE,
        shuffle=False,
        num_workers=N_DATALOADER_WORKER,
    )
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(full_unlabeled_loader)):
            outputs = model(batch)
            logits = outputs["logits"]
            scores = F.softmax(logits, dim=-1).detach().cpu().numpy()
            all_scores.append(scores)
    all_scores = np.concatenate(all_scores, axis=0)

    # estimate acc
    print(f">> estimating accuracy using {len(labeled_samples)} labeled samples")
    print(f">> estimating accuracy using {len(labeled_samples)} labeled samples")
    halfsize = len(labeled_samples) // 2
    firsthalf, secondhalf = labeled_samples[:halfsize], labeled_samples[halfsize:]
    accs = []
    for labelprop_est_samples, valid_samples in [
        (firsthalf, secondhalf),
        (secondhalf, firsthalf),
    ]:
        half_labeled_dataset = RobertaDataset(
            samples=valid_samples,
            n_classes=datadef_labeled.n_classes,
            domain_names=datadef_labeled.domain_names,
            source2labelprops=calculate_labelprops(
                labelprop_est_samples,
                datadef_labeled.n_classes,
                datadef_labeled.domain_names,
            ),
        )
        metrics = do_valid(model, half_labeled_dataset)
        accs.append(metrics["f1"])
    est_acc = sum(accs) / len(accs)

    return scores, est_acc
