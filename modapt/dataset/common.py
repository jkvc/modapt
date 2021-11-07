from os.path import dirname, exists, join

import numpy as np
from config import DATA_DIR
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
