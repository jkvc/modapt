# a dataset zoo, for quick definition / retrieval of dataset definition by name
from typing import Dict, List

from modapt.dataset.dataset_def import DatasetDefinition

_DATADEFS: Dict[str, DatasetDefinition] = {}


def get_datadef(name: str) -> DatasetDefinition:
    return _DATADEFS[name]


def get_datadef_names() -> List[str]:
    return sorted(list(_DATADEFS.keys()))


def register_datadef(name: str, datadef: DatasetDefinition):
    assert name not in _DATADEFS
    _DATADEFS[name] = datadef
