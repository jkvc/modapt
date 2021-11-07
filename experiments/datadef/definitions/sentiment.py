from os.path import join
from typing import Dict, List

import numpy as np
from config import DATA_DIR, KFOLD
from experiments.datadef.zoo import DatasetDefinition, register_datadef
from modapt.dataset.data_sample import DataSample
from modapt.utils import load_json
from tqdm import tqdm

SENTIMENT_SOURCES = [
    "airline",
    "amazon",
    "imdb",
    "senti140",
    "sst",
]
SENTIMENT_SOURCE2IDX = {c: i for i, c in enumerate(SENTIMENT_SOURCES)}


POLARITY_NAMES = ["neg", "pos"]


def polarity2idx(polarity: str) -> int:
    if polarity == "neg":
        return 0
    if polarity == "pos":
        return 1
    raise ValueError()


_LABELPROPS_DIR = join(DATA_DIR, "sentiment", "labelprops")


def load_labelprops(split):
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(_LABELPROPS_DIR, f"{split}.json")
        ).items()
    }


def load_sentiment_samples(sources: List[str], split: str) -> List[DataSample]:
    assert split in ["train", "valid", "test"]

    samples = []
    for source in tqdm(sources):
        ids = load_json(join(DATA_DIR, "sentiment", "splits", f"{source}.{split}.json"))
        raw_data = load_json(join(DATA_DIR, "sentiment", f"{source}.json"))

        for id in ids:
            samples.append(
                DataSample(
                    id=id,
                    text=raw_data[id]["text"],
                    y_idx=polarity2idx(raw_data[id]["polarity"]),
                    source_name=source,
                    source_idx=SENTIMENT_SOURCE2IDX[source],
                )
            )
    return samples


def load_splits(
    categories: List[str], splits: List[str]
) -> Dict[str, List[DataSample]]:
    return {split: load_sentiment_samples(categories, split) for split in splits}


register_datadef(
    "sentiment",
    DatasetDefinition(
        source_names=SENTIMENT_SOURCES,
        label_names=POLARITY_NAMES,
        load_splits_func=load_splits,
        load_labelprops_func=load_labelprops,
    ),
)
