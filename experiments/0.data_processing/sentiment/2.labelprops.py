from os import makedirs
from os.path import join

from modapt.datadef.definitions.sentiment import (
    _LABELPROPS_DIR,
    POLARITY_NAMES,
    SENTIMENT_SOURCES,
    load_sentiment_samples,
)
from modapt.dataset.common import calculate_labelprops
from modapt.utils import save_json

makedirs(_LABELPROPS_DIR, exist_ok=True)


for split in ["train", "valid", "test"]:
    samples = load_sentiment_samples(SENTIMENT_SOURCES, split)
    source2labelprops = calculate_labelprops(
        samples, len(POLARITY_NAMES), SENTIMENT_SOURCES
    )
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(_LABELPROPS_DIR, f"{split}.json"),
    )
