from os.path import dirname, exists, join
from typing import List

import numpy as np
import pandas as pd
from config import DATA_DIR
from modapt.dataset.data_sample import DataSample
from modapt.utils import load_json


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
            "text": sample.text,
            "y_idx": sample.y_idx,
            "domain_name": sample.domain_name,
        }
        for sample in samples
    ]
    df = pd.DataFrame(l)
    return df


def from_df(df: pd.DataFrame) -> List[DataSample]:
    ds = df.to_dict("records")
    domains = {d["domain_name"] for d in ds}
    domain2idx = {d: i for i, d in enumerate(domains)}

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
    return l
