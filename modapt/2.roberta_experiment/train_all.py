# Usage: python <script_name> <dataset_name> <n_epoch> <model_arch>

import sys
from os import makedirs
from os.path import basename, exists, join, realpath

import torch
from config import BATCHSIZE, MODELS_DIR
from modapt.datadef.zoo import get_datadef
from modapt.dataset.roberta_dataset import RobertaDataset
from modapt.eval import reduce_and_save_metrics
from modapt.experiments import run_experiments
from modapt.learning import do_valid
from modapt.model.roberta_config.base import load_roberta_model_config
from modapt.utils import DEVICE, save_json

_DATASET_NAME = sys.argv[1]
_N_TRAIN_EPOCH = int(sys.argv[2])
_ARCH = sys.argv[3]


_DATADEF = get_datadef(_DATASET_NAME)
_CONFIG = load_roberta_model_config(_ARCH, _DATADEF.n_classes, _DATADEF.n_sources)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)


train_sources = _DATADEF.source_names
train_samples = _DATADEF.load_splits_func(train_sources, ["train"])["train"]


train_dataset = RobertaDataset(
    train_samples,
    n_classes=_DATADEF.n_classes,
    source_names=_DATADEF.source_names,
    source2labelprops=_DATADEF.load_labelprops_func("train"),
)
valid_dataset = train_dataset

logdir2datasets = {
    _SAVE_DIR: {
        "train": train_dataset,
        "valid": valid_dataset,
    }
}

run_experiments(
    _CONFIG,
    logdir2datasets,
    batchsize=BATCHSIZE,
    save_model_checkpoint=True,
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
)

for e in range(_N_TRAIN_EPOCH):
    reduce_and_save_metrics(_SAVE_DIR, f"leaf_epoch_{e}.json", f"mean_epoch_{e}.json")

# run test sets for each source

checkpointpath = join(_SAVE_DIR, "checkpoint.pth")
model = torch.load(checkpointpath).to(DEVICE)

for source in _DATADEF.source_names:
    makedirs(join(_SAVE_DIR, source))
    save_metric_path = join(_SAVE_DIR, source, "leaf_test.json")
    if exists(save_metric_path):
        print(">> skip test", source)
        continue
    else:
        print(">> test", source)

    test_samples = _DATADEF.load_splits_func([source], ["test"])["test"]
    test_dataset = RobertaDataset(
        test_samples,
        n_classes=_DATADEF.n_classes,
        source_names=_DATADEF.source_names,
        source2labelprops=_DATADEF.load_labelprops_func("test"),
    )

    test_metrics = do_valid(model, test_dataset)
    save_json(test_metrics, save_metric_path)

reduce_and_save_metrics(_SAVE_DIR, f"leaf_test.json", f"mean_test.json")
