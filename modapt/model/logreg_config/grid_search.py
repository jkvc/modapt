import itertools

from modapt.model.logreg_config.base import load_logreg_model_config

_LOGREG_ARCH_PREFIX = "logreg"
_CONFIG_OVERRIDES = [
    ("+sn", "use_source_individual_norm"),
    ("+kb", "use_log_labelprop_bias"),
    ("+lr", "use_learned_residualization"),
    ("+gr", "use_gradient_reversal"),
]
# tbh these are the only ones that make sense
_ARCH_FILTER = [
    "logreg",
    "logreg+kb",
    "logreg+sn+kb",
    "logreg+gr",
    "logreg+lr",
    "logreg+sn",
    "logreg+sn+gr",
    "logreg+sn+lr",
]


def load_logreg_model_config_all_archs(n_classes, n_sources):
    base_config = load_logreg_model_config(_LOGREG_ARCH_PREFIX, n_classes, n_sources)

    arch2configs = {}
    combinations = itertools.product([False, True], repeat=len(_CONFIG_OVERRIDES))
    for comb in combinations:
        arch = _LOGREG_ARCH_PREFIX
        config_copy = {**base_config}
        for (prefix, key), value in zip(_CONFIG_OVERRIDES, comb):
            if value:
                arch += prefix
            config_copy[key] = value

        arch2configs[arch] = config_copy

    filtered = {arch: arch2configs[arch] for arch in _ARCH_FILTER}

    return filtered
