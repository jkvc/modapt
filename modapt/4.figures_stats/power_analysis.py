# Usage: python <script_name> <dataset_name> <model_arch1> <model_arch2>

import sys
from os import makedirs
from os.path import exists, join

import numpy as np
import torch
from config import BATCHSIZE, LEXICON_DIR, MODELS_DIR, OUTPUT_DIR
from modapt.datadef.zoo import get_datadef
from modapt.dataset.bow_dataset import (
    build_bow_full_batch,
    get_all_tokens,
)
from modapt.dataset.data_sample import DataSample
from modapt.dataset.roberta_dataset import RobertaDataset
from modapt.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from modapt.utils import (
    DEVICE,
    load_json,
    read_txt_as_str_list,
    save_json,
)
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

_DATASET_NAME = sys.argv[1]
_ARCH1 = sys.argv[2]
_ARCH2 = sys.argv[3]

_DATADEF = get_datadef(_DATASET_NAME)
_VALID_OR_TEST = "test"
_LOAD_SPLIT_NAME = "train" if _VALID_OR_TEST == "valid" else "test"


def _get_model_dir(arch):
    if arch.startswith("logreg"):
        return join(LEXICON_DIR, _DATASET_NAME, "holdout_source", arch)
    elif arch.startswith("roberta"):
        return join(MODELS_DIR, _DATASET_NAME, "holdout_source", arch)


def valid_roberta_model(arch):
    print(">>", arch)

    model_dir = join(MODELS_DIR, _DATASET_NAME, "holdout_source", arch)
    save_preds_dir = join(model_dir, f"{_VALID_OR_TEST}_preds")
    makedirs(save_preds_dir, exist_ok=True)

    for holdout_source in _DATADEF.source_names:
        print(">>>>", holdout_source)
        save_preds_path = join(
            model_dir, f"{_VALID_OR_TEST}_preds", f"{holdout_source}.json"
        )
        if exists(save_preds_path):
            continue

        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func([holdout_source], [_LOAD_SPLIT_NAME])[
            _LOAD_SPLIT_NAME
        ]
        valid_dataset = RobertaDataset(
            valid_samples,
            n_classes=_DATADEF.n_classes,
            source_names=_DATADEF.source_names,
            source2labelprops=_DATADEF.load_labelprops_func(_LOAD_SPLIT_NAME),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=150,
            shuffle=True,
            num_workers=6,
        )

        checkpoint_path = join(model_dir, holdout_source, "checkpoint.pth")
        model = torch.load(checkpoint_path).to(DEVICE)
        model.eval()

        id2results = {}
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                outputs = model(batch)
                logits = outputs["logits"].detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                labels = outputs["labels"].detach().cpu().numpy()
                ids = batch["id"]
                for id, pred, label in zip(ids, preds, labels):
                    id2results[id] = {
                        "pred": int(pred),
                        "label": int(label),
                        "correct": bool(pred == label),
                    }
        save_json(id2results, save_preds_path)


def valid_logreg_model(arch):
    print(">>", arch)
    config = load_logreg_model_config_all_archs(_DATADEF.n_classes, _DATADEF.n_sources)[
        arch
    ]

    model_dir = join(LEXICON_DIR, _DATASET_NAME, "holdout_source", arch)
    save_preds_dir = join(model_dir, f"{_VALID_OR_TEST}_preds")
    makedirs(save_preds_dir, exist_ok=True)

    for holdout_source in _DATADEF.source_names:
        print(">>>>", holdout_source)
        save_preds_path = join(
            model_dir, f"{_VALID_OR_TEST}_preds", f"{holdout_source}.json"
        )
        if exists(save_preds_path):
            continue

        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func([holdout_source], [_LOAD_SPLIT_NAME])[
            _LOAD_SPLIT_NAME
        ]
        model = torch.load(join(model_dir, holdout_source, "model.pth"))
        batch = build_bow_full_batch(
            valid_samples,
            _DATADEF,
            get_all_tokens(valid_samples),
            read_txt_as_str_list(join(model_dir, holdout_source, "vocab.txt")),
            use_source_individual_norm=config["use_source_individual_norm"],
            labelprop_split=_LOAD_SPLIT_NAME,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(batch)

        logits = outputs["logits"].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        labels = outputs["labels"].detach().cpu().numpy()
        ids = [s.id for s in valid_samples]

        id2results = {}
        for id, pred, label in zip(ids, preds, labels):
            id2results[id] = {
                "pred": int(pred),
                "label": int(label),
                "correct": bool(pred == label),
            }
        save_json(id2results, save_preds_path)


# run prediction on respective valid sets
for arch in [_ARCH1, _ARCH2]:
    if arch.startswith("roberta"):
        valid_roberta_model(arch)
    if arch.startswith("logreg"):
        valid_logreg_model(arch)

_OUTPUT_SAVE_DIR = join(OUTPUT_DIR, "power_analysis")
makedirs(_OUTPUT_SAVE_DIR, exist_ok=True)


# build tables
holdout_source_to_table = {}
fulltable = np.zeros((2, 2))
for holdout_source in _DATADEF.source_names:
    table = np.zeros((2, 2))
    arch1_preds = load_json(
        join(
            _get_model_dir(_ARCH1),
            f"{_VALID_OR_TEST}_preds",
            f"{holdout_source}.json",
        )
    )
    arch2_preds = load_json(
        join(
            _get_model_dir(_ARCH2),
            f"{_VALID_OR_TEST}_preds",
            f"{holdout_source}.json",
        )
    )
    ids = list(arch1_preds.keys())
    for id in ids:
        arch1_correct = arch1_preds[id]["correct"]
        arch2_correct = arch2_preds[id]["correct"]
        if arch1_correct and arch2_correct:
            table[0][0] += 1
            fulltable[0][0] += 1
        if arch1_correct and not arch2_correct:
            table[0][1] += 1
            fulltable[0][1] += 1
        if not arch1_correct and arch2_correct:
            table[1][0] += 1
            fulltable[1][0] += 1
        if not arch1_correct and not arch2_correct:
            table[1][1] += 1
            fulltable[1][1] += 1
    holdout_source_to_table[holdout_source] = table


# mcnemars
results = {}
for holdout_source in _DATADEF.source_names:
    table = holdout_source_to_table[holdout_source]
    result = mcnemar(table.T)
    results[holdout_source] = {
        "pvalue": result.pvalue,
        "statistic": result.statistic,
    }
all_result = mcnemar(fulltable.T)
results["all"] = {
    "pvalue": all_result.pvalue,
    "statistic": all_result.statistic,
}
results["fulltable"] = fulltable.tolist()

save_json(
    results, join(_OUTPUT_SAVE_DIR, f"mcnemars_{_DATASET_NAME}.{_ARCH1}@{_ARCH2}.json")
)
print("mcnemars p", results["all"]["pvalue"])

# card, power analysis
def compute_power(prob_table, dataset_size, alpha=0.05, r=5000):
    """
    Dallas Card et al. "With Little Power Comes Great Responsibility"
    https://colab.research.google.com/drive/1anaS-9ElouZhUgCAYQt8jy8qBiaXnnK1?usp=sharing#scrollTo=OCz-VAm_ifqZ
    """
    if prob_table[0, 1] == prob_table[1, 0]:
        raise RuntimeError("Power is undefined when the true effect is zero.")

    pvals = []
    diffs = []
    for i in trange(r):  # number of simulations
        sample = np.random.multinomial(
            n=dataset_size, pvals=prob_table.reshape((4,))
        ).reshape((2, 2))
        acc_diff = (sample[0, 1] - sample[1, 0]) / dataset_size
        test_results = mcnemar(sample)
        pvals.append(test_results.pvalue)
        diffs.append(acc_diff)

    true_diff = prob_table[0, 1] - prob_table[1, 0]
    true_sign = np.sign(true_diff)
    sig_diffs = [d for i, d in enumerate(diffs) if pvals[i] <= alpha]
    power = (
        len(
            [
                d
                for i, d in enumerate(diffs)
                if pvals[i] <= alpha and np.sign(d) == true_sign
            ]
        )
        / r
    )
    mean_effect = np.mean(diffs)
    type_m = np.mean(np.abs(sig_diffs) / np.abs(true_diff))
    type_s = np.mean(np.sign(sig_diffs) != true_sign)
    return power, mean_effect, type_m, type_s


results = {}
# for holdout_source in _DATADEF.source_names:
#     table = holdout_source_to_table[holdout_source]
#     dataset_size = table.sum()
#     test_set_size = len(_DATADEF.load_splits_func([holdout_source], ["test"])["test"])
#     try:
#         power, _, _, _ = compute_power(table / dataset_size, test_set_size)
#     except RuntimeError:
#         # hack bc technique can have no effect on some particular source so no power
#         power = 0
#     results[holdout_source] = {
#         "power": power,
#         "probs_table": (table / dataset_size).tolist(),
#     }

full_dataset_size = fulltable.sum()
test_set_size = len(_DATADEF.load_splits_func(_DATADEF.source_names, ["test"])["test"])
power, _, _, _ = compute_power(fulltable / full_dataset_size, test_set_size)
# power_balanced = np.array([v["power"] for v in results.values()]).mean()
# results["power_balanced"] = power_balanced
results["all"] = {
    "power": power,
    "probs_table": (fulltable / full_dataset_size).tolist(),
}
save_json(
    results, join(_OUTPUT_SAVE_DIR, f"power_{_DATASET_NAME}.{_ARCH1}@{_ARCH2}.json")
)
print("card power", results["all"]["power"])
# print("card power balanced", results["power_balanced"])
