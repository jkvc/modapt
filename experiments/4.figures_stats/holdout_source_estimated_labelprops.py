# Usage: python <script_name> <dataset_name> <lexicon_arch> <roberta_arch>

import sys
from collections import defaultdict
from os import makedirs
from os.path import exists, join
from pprint import pprint
from random import Random

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import FIGURE_DPI, FIGURES_DIR, LEXICON_DIR, MODELS_DIR, RANDOM_SEED
from experiments.datadef.zoo import get_datadef
from modapt.dataset.common import calculate_labelprops
from modapt.dataset.roberta_dataset import RobertaDataset
from modapt.learning import valid_epoch
from modapt.lexicon import eval_lexicon_model
from modapt.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from modapt.utils import (
    AUTO_DEVICE,
    load_json,
    read_txt_as_str_list,
    save_json,
    save_plt,
    stylize_model_arch_for_figures,
)
from torch.utils.data import DataLoader

_DATASET_NAME = sys.argv[1]
_LEXICON_ARCH = sys.argv[2]
_ROBERTA_ARCH = sys.argv[3]

_N_TRIALS = 5

_DATADEF = get_datadef(_DATASET_NAME)
_LEXICON_MODEL_ROOT = join(LEXICON_DIR, _DATASET_NAME, "holdout_source", _LEXICON_ARCH)
_ROBERTA_MODEL_ROOT = join(MODELS_DIR, _DATASET_NAME, "holdout_source", _ROBERTA_ARCH)
assert exists(join(_LEXICON_MODEL_ROOT, "mean_metrics.json"))
assert exists(join(_ROBERTA_MODEL_ROOT, "mean_metrics.json"))
_LEXICON_CONFIG = load_logreg_model_config_all_archs(
    _DATADEF.n_classes, _DATADEF.n_sources
)[_LEXICON_ARCH]

_SAVE_DIR = join(
    FIGURES_DIR,
    "holdout_source_estimated_labelprops",
    _DATASET_NAME,
)
makedirs(_SAVE_DIR, exist_ok=True)

_RNG = Random()
_RNG.seed(RANDOM_SEED)

_LABELPROPS_ESTIMATE_NSAMPLES = [
    100,
    150,
    200,
    250,
    300,
    350,
    400,
]

_PLOT_NROW = 3
_PLOT_NCOL = 2


def _2fold(samples):
    halfsize = len(samples) // 2
    firsthalf, secondhalf = samples[:halfsize], samples[halfsize:]
    return [[firsthalf, secondhalf], [secondhalf, firsthalf]]


# load samples, shuffle once, use this seeded shuffle order for all evals

source2samples = {}
for source in _DATADEF.source_names:
    source2samples[source] = _DATADEF.load_splits_func([source], ["train"])["train"]
    _RNG.shuffle(source2samples[source])

# lexicon model predicting with gt & estimated labelprops

_LEXICON_MODEL_PERFORMANCE_SAVE_PATH = join(_SAVE_DIR, f"{_LEXICON_ARCH}.json")

if not exists(_LEXICON_MODEL_PERFORMANCE_SAVE_PATH):
    orig_metrics = load_json(join(_LEXICON_MODEL_ROOT, "mean_metrics.json"))
    gt_source2acc = {
        source: orig_metrics[source]["mean"]["valid_f1"]
        for source in _DATADEF.source_names
    }
    gt_source2acc["mean"] = np.array(list(gt_source2acc.values())).mean()
    gt_source2acc["std"] = np.array(list(gt_source2acc.values())).std()

    notechnique_metrics = load_json(
        join(
            LEXICON_DIR, _DATASET_NAME, "holdout_source", "logreg", "mean_metrics.json"
        )
    )
    notechnique_source2acc = {
        source: notechnique_metrics[source]["mean"]["valid_f1"]
        for source in _DATADEF.source_names
    }
    notechnique_source2acc["mean"] = np.array(
        list(notechnique_source2acc.values())
    ).mean()
    notechnique_source2acc["std"] = np.array(
        list(notechnique_source2acc.values())
    ).std()

    lexicon_model_perf = {
        "gt": gt_source2acc,
        "no_technique": notechnique_source2acc,
    }

    for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:

        source2type2accs = defaultdict(lambda: defaultdict(list))
        for source in _DATADEF.source_names:
            print(">>", source, nsample)
            all_samples = source2samples[source]
            model = torch.load(join(_LEXICON_MODEL_ROOT, source, "model.pth")).to(
                AUTO_DEVICE
            )
            vocab = read_txt_as_str_list(join(_LEXICON_MODEL_ROOT, source, "vocab.txt"))

            def _eval_lex_model(label_est_samples, valid_samples) -> float:
                estimated_labelprops = {
                    "estimated": calculate_labelprops(
                        label_est_samples,
                        _DATADEF.n_classes,
                        _DATADEF.source_names,
                    )
                }
                datadef = get_datadef(_DATASET_NAME)
                datadef.load_labelprops_func = lambda _split: estimated_labelprops[
                    _split
                ]
                metrics = eval_lexicon_model(
                    model,
                    datadef,
                    valid_samples,
                    vocab,
                    use_source_individual_norm=_LEXICON_CONFIG[
                        "use_source_individual_norm"
                    ],
                    labelprop_split="estimated",  # match _load_labelprops_func()
                )
                return metrics["valid_f1"]

            for ti in range(_N_TRIALS):
                selected_sample = all_samples[ti * nsample : (ti + 1) * nsample]

                for label_est_samples, valid_samples in _2fold(selected_sample):
                    acc = _eval_lex_model(label_est_samples, valid_samples)
                    source2type2accs[source]["selected"].append(acc)

                fullacc = _eval_lex_model(selected_sample, all_samples)
                source2type2accs[source]["full"].append(fullacc)

        lexicon_model_perf[str(nsample)] = dict(source2type2accs)

    save_json(lexicon_model_perf, _LEXICON_MODEL_PERFORMANCE_SAVE_PATH)
else:
    lexicon_model_perf = load_json(_LEXICON_MODEL_PERFORMANCE_SAVE_PATH)

lexicon_model_stats = {}
for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:
    accs = []
    deltas = []
    for source in _DATADEF.source_names:
        source_technique_mean = np.array(
            lexicon_model_perf[str(nsample)][source]["full"]
        ).mean()
        source_base_mean = lexicon_model_perf["no_technique"][source]
        accs.append(source_technique_mean)
        deltas.append(source_technique_mean - source_base_mean)
    lexicon_model_stats[str(nsample)] = {
        "mean": np.array(accs).mean(),
        "delta_over_base": np.array(deltas).mean(),
        "std_delta_over_base": np.array(deltas).std(),
    }
save_json(lexicon_model_stats, join(_SAVE_DIR, f"{_LEXICON_ARCH}_deltas.json"))

# roberta model predicting with gt and estimated labelprops

_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH = join(_SAVE_DIR, f"{_ROBERTA_ARCH}.json")

if not exists(_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH):
    orig_metrics = load_json(join(_ROBERTA_MODEL_ROOT, "mean_metrics.json"))
    gt_source2acc = {
        source: orig_metrics[source]["mean"]["valid_f1.best"]
        for source in _DATADEF.source_names
    }
    gt_source2acc["mean"] = np.array(list(gt_source2acc.values())).mean()
    gt_source2acc["std"] = np.array(list(gt_source2acc.values())).std()

    notechnique_metrics = load_json(
        join(
            MODELS_DIR, _DATASET_NAME, "holdout_source", "roberta", "mean_metrics.json"
        )
    )
    notechnique_source2acc = {
        source: notechnique_metrics[source]["mean"]["valid_f1.best"]
        for source in _DATADEF.source_names
    }
    notechnique_source2acc["mean"] = np.array(
        list(notechnique_source2acc.values())
    ).mean()
    notechnique_source2acc["std"] = np.array(
        list(notechnique_source2acc.values())
    ).std()

    roberta_model_perf = {
        "gt": gt_source2acc,
        "no_technique": notechnique_source2acc,
    }

    for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:

        source2type2accs = defaultdict(lambda: defaultdict(list))
        for source in _DATADEF.source_names:
            print(">>", source, nsample)

            model = torch.load(join(_ROBERTA_MODEL_ROOT, source, "checkpoint.pth")).to(
                AUTO_DEVICE
            )
            all_samples = source2samples[source]

            def _eval_roberta_model(label_est_samples, valid_samples) -> float:
                estimated_labelprops = calculate_labelprops(
                    label_est_samples,
                    _DATADEF.n_classes,
                    _DATADEF.source_names,
                )
                valid_loader = DataLoader(
                    RobertaDataset(
                        valid_samples,
                        _DATADEF.n_classes,
                        _DATADEF.source_names,
                        source2labelprops=estimated_labelprops,
                    ),
                    batch_size=100,
                    shuffle=False,
                    num_workers=1,
                )
                metrics = valid_epoch(model, valid_loader)
                return metrics["f1"]

            for ti in range(_N_TRIALS):
                selected_sample = all_samples[ti * nsample : (ti + 1) * nsample]

                for label_est_samples, valid_samples in _2fold(selected_sample):
                    acc = _eval_roberta_model(label_est_samples, valid_samples)
                    source2type2accs[source]["selected"].append(acc)

                fullacc = _eval_roberta_model(selected_sample, all_samples)
                source2type2accs[source]["full"].append(fullacc)

        roberta_model_perf[str(nsample)] = source2type2accs
    save_json(roberta_model_perf, _ROBERTA_MODEL_PERFORMANCE_SAVE_PATH)
else:
    roberta_model_perf = load_json(_ROBERTA_MODEL_PERFORMANCE_SAVE_PATH)

_PLOT_SAVE_DIR = join(_SAVE_DIR, f"{_LEXICON_ARCH}@{_ROBERTA_ARCH}")
makedirs(_PLOT_SAVE_DIR, exist_ok=True)


roberta_model_stats = {}
for nsample in _LABELPROPS_ESTIMATE_NSAMPLES:
    accs = []
    deltas = []
    for source in _DATADEF.source_names:
        source_technique_mean = np.array(
            roberta_model_perf[str(nsample)][source]["full"]
        ).mean()
        source_base_mean = roberta_model_perf["no_technique"][source]
        accs.append(source_technique_mean)
        deltas.append(source_technique_mean - source_base_mean)
    roberta_model_stats[str(nsample)] = {
        "mean": np.array(accs).mean(),
        "delta_over_base": np.array(deltas).mean(),
        "std_delta_over_base": np.array(deltas).std(),
    }
save_json(roberta_model_stats, join(_SAVE_DIR, f"{_ROBERTA_ARCH}_deltas.json"))

# full acc

plt.clf()
plt.figure(figsize=(7, 5))
# roberta
plt.axhline(
    roberta_model_perf["gt"]["mean"],
    color="teal",
    linestyle="--",
    label=f"{stylize_model_arch_for_figures(_ROBERTA_ARCH)} w/ oracle label distribution",
)
plt.plot(
    _LABELPROPS_ESTIMATE_NSAMPLES,
    [
        np.array(
            [
                roberta_model_perf[str(nsample)][source]["full"]
                for source in _DATADEF.source_names
            ]
        ).mean()
        for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
    ],
    marker="D",
    c="teal",
    label=f"{stylize_model_arch_for_figures(_ROBERTA_ARCH)} w/ estimated label distribution",
)
plt.axhline(
    roberta_model_perf["no_technique"]["mean"],
    color="deepskyblue",
    linestyle="--",
    label=stylize_model_arch_for_figures("roberta"),
)
# lexicon
plt.axhline(
    lexicon_model_perf["gt"]["mean"],
    color="firebrick",
    linestyle="--",
    label=f"{stylize_model_arch_for_figures(_LEXICON_ARCH)} w/ oracle label distribution",
)
plt.plot(
    _LABELPROPS_ESTIMATE_NSAMPLES,
    [
        np.array(
            [
                lexicon_model_perf[str(nsample)][source]["full"]
                for source in _DATADEF.source_names
            ]
        ).mean()
        for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
    ],
    marker="D",
    c="firebrick",
    label=f"{stylize_model_arch_for_figures(_LEXICON_ARCH)} w/ estimated label distribution",
)
plt.axhline(
    lexicon_model_perf["no_technique"]["mean"],
    color="orange",
    linestyle="--",
    label=stylize_model_arch_for_figures("logreg"),
)

# plt.title(f"holdout source full acc under estimated labelprops ({_DATASET_NAME})")
plt.legend()
plt.xlabel("# Samples for label distribution estimation")
plt.ylabel("Holdout source accuracy")
save_plt(join(_PLOT_SAVE_DIR, "full_acc.png"))

# per source acc


def plot_single_source(ax, source):
    ax.plot(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        [
            np.array([roberta_model_perf[str(nsample)][source]["full"]]).mean()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ],
        marker="D",
        c="teal",
        label=f"{stylize_model_arch_for_figures(_ROBERTA_ARCH)} true accuracy",
    )
    # roberta line with +-std region, select valid accs
    means = np.array(
        [
            np.array([roberta_model_perf[str(nsample)][source]["selected"]]).mean()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ]
    )
    stds = np.array(
        [
            np.array([roberta_model_perf[str(nsample)][source]["selected"]]).std()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ]
    )
    ax.plot(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        means,
        c="deepskyblue",
        marker="o",
        linestyle="--",
        label=f"{stylize_model_arch_for_figures(_ROBERTA_ARCH)} estimated accuracy",
    )
    ax.fill_between(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        means - stds,
        means + stds,
        color="azure",
    )
    # lexicon single line, full valid accs
    ax.plot(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        [
            np.array([lexicon_model_perf[str(nsample)][source]["full"]]).mean()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ],
        marker="D",
        c="firebrick",
        label=f"{stylize_model_arch_for_figures(_LEXICON_ARCH)} true accuracy",
    )
    # lexicon line with +-std region, select valid accs
    means = np.array(
        [
            np.array([lexicon_model_perf[str(nsample)][source]["selected"]]).mean()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ]
    )
    stds = np.array(
        [
            np.array([lexicon_model_perf[str(nsample)][source]["selected"]]).std()
            for nsample in _LABELPROPS_ESTIMATE_NSAMPLES
        ]
    )
    ax.plot(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        means,
        marker="o",
        linestyle="--",
        c="goldenrod",
        label=f"{stylize_model_arch_for_figures(_LEXICON_ARCH)} estimated accuracy",
    )
    ax.fill_between(
        _LABELPROPS_ESTIMATE_NSAMPLES,
        means - stds,
        means + stds,
        color="cornsilk",
    )
    # ax.title(
    #     f"holdout source accs under estimated labelprops ({_DATASET_NAME}:{source})"
    # )
    ax.legend()
    ax.set_xlabel("# Samples for labelprops estimation")
    ax.set_ylabel("Holdout source accuracy")


# each source single image

for source in _DATADEF.source_names:
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_single_source(ax, source)
    save_plt(join(_PLOT_SAVE_DIR, f"compare_{source}.png"))

# all sources in a grid

plt.clf()
fig, axes = plt.subplots(nrows=_PLOT_NROW, ncols=_PLOT_NCOL, figsize=(16, 16))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(_DATADEF.source_names):
        plt.delaxes(ax)
    else:
        source = _DATADEF.source_names[i]
        print(source)
        plot_single_source(ax, source)
        ax.title.set_text(source)
save_plt(join(_PLOT_SAVE_DIR, "all_compare.png"))
