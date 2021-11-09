from os import makedirs
from os.path import dirname, exists, join
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from ..utils import load_json, save_json
from .common import calculate_labelprops, get_labelprops_full_split
from .data_sample import DataSample

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


class RobertaDataset(Dataset):
    def __init__(
        self,
        samples: List[DataSample],
        n_classes: int,
        domain_names: List[str],
        source2labelprops: Optional[Dict[str, np.array]] = None,
    ):
        self.samples: List[DataSample] = samples

        if source2labelprops is not None:
            self.source2labelprops = source2labelprops
        else:
            # estimate labelprops in given samples
            self.source2labelprops = calculate_labelprops(
                samples, n_classes, domain_names
            )

        self.tokenizer = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.tokenizer:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        sample = self.samples[idx]
        x = np.array(
            self.tokenizer.encode(
                sample.text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
        )

        return {
            "id": sample.id,
            "x": x,
            "y": sample.y_idx if sample.y_idx is not None else -1,
            "labelprops": self.source2labelprops[sample.domain_name],
            "domain_idx": sample.domain_idx,
        }


# def get_kfold_primary_frames_roberta_datasets(
#     issues: List[str],
# ) -> List[Dict[str, List[RobertaDataset]]]:
#     fold2split2samples = load_kfold_framing_samples(issues, task="primary_frame")
#     return fold2split2samples_to_roberta_datasets(fold2split2samples)


# def fold2split2samples_to_roberta_datasets(fold2split2samples):
#     fold2split2datasets = [
#         {
#             split_name: RobertaDataset(split_samples)
#             for split_name, split_samples in split2samples.items()
#         }
#         for split2samples in fold2split2samples
#     ]
#     return fold2split2datasets
