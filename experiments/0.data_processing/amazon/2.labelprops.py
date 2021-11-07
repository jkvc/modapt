from os import makedirs
from os.path import join

from modapt.datadef.definitions.amazon import (
    CATEGORIES,
    LABELPROPS_DIR,
    RATING_N_CLASSES,
)
from modapt.dataset.amazon.samples import load_all_amazon_review_samples
from modapt.dataset.common import calculate_labelprops
from modapt.utils import save_json

makedirs(LABELPROPS_DIR, exist_ok=True)

for split in ["train", "valid", "test"]:
    samples = load_all_amazon_review_samples(CATEGORIES, split)
    source2labelprops = calculate_labelprops(samples, RATING_N_CLASSES, CATEGORIES)
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(LABELPROPS_DIR, f"{split}.json"),
    )
