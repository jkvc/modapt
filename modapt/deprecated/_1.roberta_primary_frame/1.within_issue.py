import sys
from os.path import join

import modapt.models_roberta  # noqa
from config import BATCHSIZE, FOLDS_TO_RUN, ISSUES, KFOLD, MODELS_DIR
from modapt.dataset import get_kfold_primary_frames_datasets
from modapt.eval import reduce_and_save_metrics
from modapt.experiments import run_experiments

_arch = sys.argv[1]
EXPERIMENT_NAME = f"1.{_arch}"


def _train():
    path2datasets = {}
    for issue in ISSUES:
        kfold_datasets = get_kfold_primary_frames_datasets([issue])
        for ki in FOLDS_TO_RUN:
            datasets = kfold_datasets[ki]
            path2datasets[
                join(MODELS_DIR, EXPERIMENT_NAME, issue, f"fold_{ki}")
            ] = datasets
    run_experiments(_arch, path2datasets, batchsize=BATCHSIZE, save_model=False)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
