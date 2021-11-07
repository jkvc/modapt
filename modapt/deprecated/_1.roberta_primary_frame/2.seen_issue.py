import sys
from os.path import join

import modapt.models_roberta  # noqa
from config import BATCHSIZE, FOLDS_TO_RUN, ISSUES, MODELS_DIR
from modapt.dataset import get_kfold_primary_frames_datasets
from modapt.eval import reduce_and_save_metrics
from modapt.experiments import run_experiments

_arch = sys.argv[1]
EXPERIMENT_NAME = f"2.{_arch}"


def _train():
    path2datasets = {}
    kfold_datasets = get_kfold_primary_frames_datasets(ISSUES)
    for ki in FOLDS_TO_RUN:
        path2datasets[join(MODELS_DIR, EXPERIMENT_NAME, f"fold_{ki}")] = kfold_datasets[
            ki
        ]
    run_experiments(_arch, path2datasets, batchsize=BATCHSIZE)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
