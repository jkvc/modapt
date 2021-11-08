from dataclasses import dataclass
from typing import Optional


@dataclass
class DataSample:
    id: str
    text: str
    y_idx: Optional[int]
    domain_name: str
    domain_idx: int
