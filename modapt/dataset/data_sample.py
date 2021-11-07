from dataclasses import dataclass


@dataclass
class DataSample:
    id: str
    text: str
    y_idx: int
    domain_name: str
    domain_idx: int
