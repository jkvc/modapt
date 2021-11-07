from os import makedirs
from os.path import join

from modapt.dataset.arxiv.definition import (
    ARXIV_CATEGORIES,
    LABELPROPS_DIR,
    YEARRANGE_N_CLASSES,
)
from modapt.dataset.arxiv.samples import (
    load_all_arxiv_abstract_samples,
)
from modapt.dataset.common import calculate_labelprops
from modapt.utils import save_json

makedirs(LABELPROPS_DIR, exist_ok=True)

for split in ["train", "valid", "test"]:
    samples = load_all_arxiv_abstract_samples(ARXIV_CATEGORIES, split)
    source2labelprops = calculate_labelprops(
        samples, YEARRANGE_N_CLASSES, ARXIV_CATEGORIES
    )
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(LABELPROPS_DIR, f"{split}.json"),
    )
