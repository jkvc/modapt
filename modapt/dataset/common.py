from os.path import dirname, exists, join
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils import load_json
from .data_sample import DataSample
from .dataset_def import DatasetDefinition


def calculate_labelprops(samples, n_classes, domain_names):
    source2labelcounts = {
        source: (np.zeros((n_classes,)) + 1e-8) for source in domain_names
    }
    for s in samples:
        source2labelcounts[domain_names[s.domain_idx]][s.y_idx] += 1
    return {
        source: labelcounts / (labelcounts.sum())
        for source, labelcounts in source2labelcounts.items()
    }


def get_labelprops_full_split(labelprops_dir, split):
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(labelprops_dir, f"{split}.json")
        ).items()
    }


def to_df(samples: List[DataSample]) -> pd.DataFrame:
    l = [
        {
            "domain_name": sample.domain_name,
            "y_idx": sample.y_idx,
            "text": sample.text,
        }
        for sample in samples
    ]
    df = pd.DataFrame(l)
    return df


def from_labeled_df(df: pd.DataFrame) -> DatasetDefinition:
    """convert a dataframe of labeled samples to a proper dataset definition

    Args:
        df (pd.DataFrame): dataframe of labeled samples, must contain the following columns
            - text
            - y_idx
            - domain_name

    Returns:
        DatasetDefinition: the populated dataset definition
    """
    assert "y_idx" in df.columns

    ds = df.to_dict("records")

    domain_names = list({d["domain_name"] for d in ds})
    domain2idx = {d: i for i, d in enumerate(domain_names)}
    nclasses = max({d["y_idx"] for d in ds}) + 1

    l = [
        DataSample(
            id=i,
            text=d["text"],
            y_idx=d["y_idx"],
            domain_name=d["domain_name"],
            domain_idx=domain2idx[d["domain_name"]],
        )
        for i, d in enumerate(ds)
    ]
    domain2samples = {}
    for sample in l:
        if sample.domain_name not in domain2samples:
            domain2samples[sample.domain_name] = []
        domain2samples[sample.domain_name].append(sample)

    domain2labelprops = calculate_labelprops(l, nclasses, domain_names)

    def _load_splits_func(domains, _):
        samples = []
        for d in domains:
            samples.extend(domain2samples[d])
        return {"default": samples}

    def _load_labelprops_func(_):
        return domain2labelprops

    datadef = DatasetDefinition(
        domain_names=domain_names,
        label_names=[str(i) for i in range(nclasses)],
        load_splits_func=_load_splits_func,
        load_labelprops_func=_load_labelprops_func,
    )
    return datadef


def from_unlabeled_df(df: pd.DataFrame, nclasses: int) -> DatasetDefinition:
    """convert a dataframe of unlabeled samples to a proper dataset definition

    Args:
        df (pd.DataFrame): dataframe of labeled samples, must contain the following columns
            - text
            - domain_name

    Returns:
        DatasetDefinition: the populated dataset definition
    """
    ds = df.to_dict("records")

    domain_names = list({d["domain_name"] for d in ds})
    domain2idx = {d: i for i, d in enumerate(domain_names)}

    l = [
        DataSample(
            id=i,
            text=d["text"],
            y_idx=None,
            domain_name=d["domain_name"],
            domain_idx=domain2idx[d["domain_name"]],
        )
        for i, d in enumerate(ds)
    ]
    domain2samples = {}
    for sample in l:
        if sample.domain_name not in domain2samples:
            domain2samples[sample.domain_name] = []
        domain2samples[sample.domain_name].append(sample)

    def _load_splits_func(domains, _):
        samples = []
        for d in domains:
            samples.extend(domain2samples[d])
        return {"default": samples}

    def _load_labelprops_func(_):
        raise NotImplementedError("load_labelprops on dataset without labels")

    datadef = DatasetDefinition(
        domain_names=domain_names,
        label_names=[str(i) for i in range(nclasses)],
        load_splits_func=_load_splits_func,
        load_labelprops_func=_load_labelprops_func,
    )
    return datadef
