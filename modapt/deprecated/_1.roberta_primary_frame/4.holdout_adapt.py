import sys
from os.path import exists, join
from random import Random

from config import (
    BATCHSIZE,
    DATASET_SIZES,
    FOLDS_TO_RUN,
    ISSUES,
    KFOLD,
    MODELS_DIR,
    RANDOM_SEED,
)
from modapt.dataset import PrimaryFrameDataset
from modapt.eval import reduce_and_save_metrics, reduce_tree_inplace
from modapt.experiments import run_experiments
from modapt.text_samples import load_kfold_text_samples
from modapt.utils import load_json, save_json
from modapt.viualization import visualize_num_sample_num_epoch

RNG = Random()


_arch = sys.argv[1]

EXPERIMENT_NAME = f"4.{_arch}"
CHECKPOINT_EXPERIMENT_NAME = f"3f.{_arch}"

MAX_EPOCH = 10


def _train():
    # root/numsample/holdout_issue/fold
    path2datasets = {}
    path2checkpointpath = {}

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"

        fold2split2samples = load_kfold_text_samples(
            [holdout_issue], task="primary_frame"
        )
        for ki in FOLDS_TO_RUN:
            split2samples = fold2split2samples[ki]
            RNG.seed(RANDOM_SEED)
            RNG.shuffle(split2samples["train"])

            for numsample in DATASET_SIZES:
                train_samples = split2samples["train"][:numsample]
                valid_samples = split2samples["valid"]
                train_dataset = PrimaryFrameDataset(
                    train_samples, labelprops_source="train"
                )
                valid_dataset = PrimaryFrameDataset(
                    valid_samples, labelprops_source="train"
                )

                save_dir = join(
                    MODELS_DIR,
                    EXPERIMENT_NAME,
                    f"{numsample:04}_samples",
                    model_name,
                    f"fold_{ki}",
                )
                path2datasets[save_dir] = {
                    "train": train_dataset,
                    "valid": valid_dataset,
                }
                path2checkpointpath[save_dir] = join(
                    MODELS_DIR,
                    CHECKPOINT_EXPERIMENT_NAME,
                    model_name,
                    "checkpoint.pth",
                )

    run_experiments(
        _arch,
        path2datasets,
        path2checkpointpath=path2checkpointpath,
        save_model=False,
        keep_latest=True,
        batchsize=BATCHSIZE,
        max_epochs=MAX_EPOCH,
        skip_train_zeroth_epoch=True,
    )


def calculate_best_earlystop_metrics():
    model_root = join(MODELS_DIR, EXPERIMENT_NAME)
    numepoch2metrics = {
        epoch: load_json(join(model_root, f"mean_epoch_{epoch}.json"))
        for epoch in range(MAX_EPOCH)
        if exists(join(model_root, f"mean_epoch_{epoch}.json"))
    }
    bestearlystop_metrics = {}

    for numsample in DATASET_SIZES:
        bestearlystop_metrics[numsample] = {}
        for issue in ISSUES:
            bestearlystop_metrics[numsample][issue] = {}
            for ki in FOLDS_TO_RUN:
                valids = []
                for numepoch in range(MAX_EPOCH):
                    if numepoch not in numepoch2metrics:
                        continue
                    valid = numepoch2metrics[numepoch][f"{numsample:04}_samples"][
                        f"holdout_{issue}"
                    ][f"fold_{ki}"]["mean"]["valid_f1"]
                    valids.append(valid)
                best_valid = max(valids)
                bestearlystop_metrics[numsample][issue][ki] = {
                    "mean": {"best_earlystop_valid_f1": best_valid}
                }

    reduce_tree_inplace(bestearlystop_metrics)
    save_json(bestearlystop_metrics, join(model_root, "best_earlystop.json"))


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    for epoch in range(MAX_EPOCH):
        reduce_and_save_metrics(
            join(MODELS_DIR, EXPERIMENT_NAME),
            leaf_metric_filename=f"leaf_epoch_{epoch}.json",
            save_filename=f"mean_epoch_{epoch}.json",
        )

    visualize_num_sample_num_epoch(
        join(MODELS_DIR, EXPERIMENT_NAME),
        DATASET_SIZES,
        range(MAX_EPOCH),
        title=EXPERIMENT_NAME,
        legend_title="num samples",
        xlabel="epoch idx",
        ylabel="valid f1",
    )

    calculate_best_earlystop_metrics()
