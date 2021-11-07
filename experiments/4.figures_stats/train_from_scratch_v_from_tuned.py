# Usage: python <script_name> <dataset_name> <roberta_arch>


import sys
from os import makedirs
from os.path import join
from random import Random

import matplotlib.pyplot as plt
import numpy as np
from config import FIGURES_DIR, MODELS_DIR, ROBERTA_ADAPT_N_SAMPLES
from experiments.datadef.zoo import get_datadef
from modapt.utils import load_json, save_plt

_DATASET_NAME = sys.argv[1]
_ROBERTA_ARCH = sys.argv[2]

_SAVE_DIR = join(
    FIGURES_DIR,
    "train_from_scratch_v_from_tuned",
    _DATASET_NAME,
)
makedirs(_SAVE_DIR, exist_ok=True)

_DATADEF = get_datadef(_DATASET_NAME)
_HOLDOUT_SOUCE_MODEL_ROOT = join(
    MODELS_DIR, _DATASET_NAME, "holdout_source", _ROBERTA_ARCH
)
_HOLDOUT_ADAPT_MODEL_ROOT = join(
    MODELS_DIR, _DATASET_NAME, "holdout_adapt", _ROBERTA_ARCH
)
_FROM_SCRATCH_MODEL_ROOT = join(
    MODELS_DIR, _DATASET_NAME, "single_source_from_scratch", _ROBERTA_ARCH
)


def load_metrics(model_root):
    nsample2acc = {}
    nsample2accs = {}
    metrics = load_json(join(model_root, "mean_metrics.json"))
    for nsample in ROBERTA_ADAPT_N_SAMPLES:
        accs = []
        for source in _DATADEF.domain_names:
            accs.append(
                metrics[f"{nsample:04}_samples"][source]["mean"]["valid_f1.best"]
            )
        nsample2accs[nsample] = np.array(accs)
        nsample2acc[nsample] = np.array(accs).mean()
    return nsample2acc, nsample2accs


holdout_adapt_nsample2acc, holdout_adapt_nsample2accs = load_metrics(
    _HOLDOUT_ADAPT_MODEL_ROOT
)
from_scratch_nsample2acc, from_scratch_nsample2accs = load_metrics(
    _FROM_SCRATCH_MODEL_ROOT
)

holdout_source_metrics = load_json(join(_HOLDOUT_SOUCE_MODEL_ROOT, "mean_metrics.json"))
holdout_source_acc = np.array(
    [
        holdout_source_metrics[source]["mean"]["valid_f1.best"]
        for source in _DATADEF.domain_names
    ]
).mean()


_PLOT_SAVE_PATH = join(_SAVE_DIR, f"{_ROBERTA_ARCH}.png")
plt.clf()
plt.figure(figsize=(7, 5))
plt.plot(
    ROBERTA_ADAPT_N_SAMPLES,
    [holdout_adapt_nsample2acc[nsample] for nsample in ROBERTA_ADAPT_N_SAMPLES],
    marker="D",
    c="teal",
    label=f"Fine-tuned from pretrained",
)
# for nsample in ROBERTA_ADAPT_N_SAMPLES:
#     plt.scatter(
#         np.ones((len(_DATADEF.domain_names),)) * nsample,
#         holdout_adapt_nsample2accs[nsample],
#         edgecolors="teal",
#         facecolors="none",
#         marker="D",
#         s=12,
#     )

plt.axhline(
    holdout_source_acc,
    color="teal",
    linestyle="--",
    label=f"No fine-tuning",
)
plt.plot(
    ROBERTA_ADAPT_N_SAMPLES,
    [from_scratch_nsample2acc[nsample] for nsample in ROBERTA_ADAPT_N_SAMPLES],
    marker="D",
    c="chocolate",
    label=f"Fine-tuned from off-the-shelf",
)
# for nsample in ROBERTA_ADAPT_N_SAMPLES:
#     plt.scatter(
#         np.ones((len(_DATADEF.domain_names),)) * nsample,
#         from_scratch_nsample2accs[nsample],
#         edgecolors="chocolate",
#         facecolors="none",
#         marker="D",
#         s=12,
#     )
# plt.title(f"impact of fine-tuning ({_DATASET_NAME})")
plt.legend()
plt.xlabel("# Samples for adaptation")
plt.ylabel("Holdout source accuracy")
save_plt(_PLOT_SAVE_PATH)
