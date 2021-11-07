from os import mkdir
from os.path import exists, join

import pandas as pd
import torch
import torch.nn as nn
from config import ISSUES, MODELS_DIR

from modapt import models
from modapt.data_aug import (
    augment_train_splits,
    get_kfold_single_span_frame_train_samples,
)
from modapt.dataset import (
    fold2split2samples_to_datasets,
    load_kfold_primary_frame_samples,
)
from modapt.eval import reduce_and_save_metrics
from modapt.learning import get_kfold_metrics, train
from modapt.utils import mkdir_overwrite, write_str_list_as_txt

# EXPERIMENT_NAME = "2.0.1.2.meddrop_half.w2.min150"
# MIN_SPAN_LEN = 150
# ARCH = "roberta_meddrop_half"
# AUG_WEIGHT = 0.2
# BATCHSIZE = 50

EXPERIMENT_NAME = "2.0.1.2.meddrop.w2.min150"
MIN_SPAN_LEN = 150
ARCH = "roberta_meddrop"
AUG_WEIGHT = 0.2
BATCHSIZE = 25

FOLDS_TO_RUN = [0, 1, 2]
KFOLD = 8


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    # kfold_datasets = get_kfold_primary_frames_datasets(ISSUES, KFOLD)
    fold2split2samples = load_kfold_primary_frame_samples(ISSUES, KFOLD)
    print(">> before aug", len(fold2split2samples[0]["train"]))
    aug_fold2samples = get_kfold_single_span_frame_train_samples(
        ISSUES, KFOLD, MIN_SPAN_LEN, AUG_WEIGHT
    )
    augment_train_splits(fold2split2samples, aug_fold2samples)
    print(">>  after aug", len(fold2split2samples[0]["train"]))

    augmented_datasets = fold2split2samples_to_datasets(fold2split2samples)
    for ki, datasets in enumerate(augmented_datasets):
        if ki not in FOLDS_TO_RUN:
            print(">> not running fold", ki)
            continue

        save_fold = join(save_root, f"fold_{ki}")
        if exists(join(save_fold, "_complete")):
            print(">> skip", ki)
            continue
        mkdir_overwrite(save_fold)

        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]

        model = models.get_model(ARCH)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train(
            model,
            train_dataset,
            valid_dataset,
            save_fold,
            batchsize=BATCHSIZE,
        )

        write_str_list_as_txt(["."], join(save_fold, "_complete"))


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    # _valid_combined_issue()
    # _valid_individual_issue()

# def _valid_combined_issue():
#     experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
#     assert exists(
#         experiment_root_path
#     ), f"{experiment_root_path} does not exist, choose the correct experiment name"

#     metrics_save_filepath = join(experiment_root_path, "metrics_combined_issues.csv")
#     assert not exists(metrics_save_filepath)

#     metrics = get_kfold_metrics(
#         ISSUES,
#         KFOLD,
#         experiment_root_path,
#         zeroth_fold_only=ZEROTH_FOLD_ONLY,
#     )
#     metrics = {"all": metrics}

#     df = pd.DataFrame.from_dict(metrics, orient="index")
#     print(df)
#     df.to_csv(metrics_save_filepath)


# def _valid_individual_issue():
#     experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
#     assert exists(
#         experiment_root_path
#     ), f"{experiment_root_path} does not exist, choose the correct experiment name"

#     metrics_save_filepath = join(experiment_root_path, "metrics_individual_issues.csv")
#     assert not exists(metrics_save_filepath)

#     issue2metrics = {}
#     for issue in ISSUES:
#         print(issue)
#         metrics = get_kfold_metrics(
#             [issue],
#             KFOLD,
#             experiment_root_path,
#             zeroth_fold_only=ZEROTH_FOLD_ONLY,
#         )
#         issue2metrics[issue] = metrics

#     df = pd.DataFrame.from_dict(issue2metrics, orient="index")
#     df.loc["mean"] = df.mean()
#     print(df)
#     df.to_csv(metrics_save_filepath)
