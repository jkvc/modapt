import sys
from collections import defaultdict
from os import makedirs
from os.path import join
from pprint import pprint
from random import Random

import modapt.models_roberta  # noqa
from config import BATCHSIZE, ISSUES, MODELS_DIR
from modapt.dataset import PrimaryFrameDataset
from modapt.eval import reduce_and_save_metrics
from modapt.experiments import run_experiments
from modapt.text_samples import load_all_text_samples

_arch = sys.argv[1]
EXPERIMENT_NAME = f"6f.{_arch}"


def _train():
    path2datasets = {}

    for issue in ISSUES:
        train_issues_all_samples = load_all_text_samples(
            [issue],
            split="train",
            task="primary_frame",
        )
        train_issue_dataset = PrimaryFrameDataset(train_issues_all_samples)

        valid_issues = [i for i in ISSUES if i != issue]
        holdout_issue_all_samples = load_all_text_samples(
            valid_issues,
            split="train",
            task="primary_frame",
        )
        holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

        path2datasets[join(MODELS_DIR, EXPERIMENT_NAME, issue)] = {
            "train": train_issue_dataset,
            "valid": holdout_issue_dataset,
        }
    run_experiments(
        _arch,
        path2datasets,
        batchsize=BATCHSIZE,
        num_early_stop_non_improve_epoch=4,
    )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
