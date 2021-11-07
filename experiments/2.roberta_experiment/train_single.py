# Usage: python <script_name> <dataset_name> <n_epoch> <model_arch>

import sys
from os.path import basename, exists, join, realpath
from random import Random

import torch
from config import BATCHSIZE, MODELS_DIR, RANDOM_SEED
from experiments.datadef.zoo import get_datadef
from modapt.dataset.roberta_dataset import RobertaDataset
from modapt.eval import reduce_and_save_metrics
from modapt.experiments import run_experiments
from modapt.learning import do_valid
from modapt.model.roberta_config.base import load_roberta_model_config
from modapt.utils import AUTO_DEVICE, save_json

_DATASET_NAME = sys.argv[1]
_N_TRAIN_EPOCH = int(sys.argv[2])
_ARCH = sys.argv[3]


_DATADEF = get_datadef(_DATASET_NAME)
_CONFIG = load_roberta_model_config(_ARCH, _DATADEF.n_classes, _DATADEF.n_sources)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)

_RNG = Random()
_RNG.seed(RANDOM_SEED)

logdir2datasets = {}

for train_source in _DATADEF.domain_names:
    print(">>", train_source)

    train_samples = _DATADEF.load_splits_func([train_source], ["train"])["train"]

    valid_sources = [s for s in _DATADEF.domain_names if s != train_source]
    sourcename2validsamples = {
        sourcename: _DATADEF.load_splits_func([sourcename], ["train"])["train"]
        for sourcename in valid_sources
    }
    n_valid_samples_per_source = min(
        len(samples) for samples in sourcename2validsamples.values()
    )
    print(n_valid_samples_per_source)
    valid_samples = []
    for domain_name in valid_sources:
        all_samples_from_source = sourcename2validsamples[domain_name]
        _RNG.shuffle(all_samples_from_source)
        valid_samples.extend(all_samples_from_source[:n_valid_samples_per_source])

    train_dataset = RobertaDataset(
        train_samples,
        n_classes=_DATADEF.n_classes,
        domain_names=_DATADEF.domain_names,
        source2labelprops=_DATADEF.load_labelprops_func("train"),
    )
    valid_dataset = RobertaDataset(
        valid_samples,
        n_classes=_DATADEF.n_classes,
        domain_names=_DATADEF.domain_names,
        source2labelprops=_DATADEF.load_labelprops_func("train"),
    )

    logdir2datasets[join(_SAVE_DIR, train_source)] = {
        "train": train_dataset,
        "valid": valid_dataset,
    }


run_experiments(
    _CONFIG,
    logdir2datasets,
    batchsize=BATCHSIZE,
    save_model_checkpoint=True,
    # train a fixed number of epoch since we can't use holdout issue as
    # validation data to early stop
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
)

reduce_and_save_metrics(_SAVE_DIR)
for e in range(_N_TRAIN_EPOCH):
    reduce_and_save_metrics(_SAVE_DIR, f"leaf_epoch_{e}.json", f"mean_epoch_{e}.json")


# setup and run test set

for train_source in _DATADEF.domain_names:
    save_metric_path = join(_SAVE_DIR, train_source, "leaf_test.json")
    if exists(save_metric_path):
        print(">> skip test", train_source)
        continue
    else:
        print(">> test except", train_source)

    test_sources = [s for s in _DATADEF.domain_names if s != train_source]
    test_samples = _DATADEF.load_splits_func(test_sources, ["test"])["test"]
    test_dataset = RobertaDataset(
        test_samples,
        n_classes=_DATADEF.n_classes,
        domain_names=_DATADEF.domain_names,
        source2labelprops=_DATADEF.load_labelprops_func("test"),
    )

    checkpointpath = join(_SAVE_DIR, train_source, "checkpoint.pth")
    model = torch.load(checkpointpath).to(AUTO_DEVICE)
    test_metrics = do_valid(model, test_dataset)

    save_json(test_metrics, save_metric_path)

reduce_and_save_metrics(_SAVE_DIR, f"leaf_test.json", f"mean_test.json")
