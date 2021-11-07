from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from modapt.dataset.data_sample import DataSample


@dataclass
class DatasetDefinition:
    domain_names: List[str]
    label_names: List[str]
    load_splits_func: Callable[[List[str], List[str]], Dict[str, List[DataSample]]]
    load_labelprops_func: Callable[[str], Dict[str, np.array]]

    n_sources: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_sources = len(self.domain_names)
        self.n_classes = len(self.label_names)
