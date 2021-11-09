from os.path import dirname, exists, join, realpath

from ...utils import load_json

VOCAB_SIZE = 5000

_logreg_config_dir = dirname(realpath(__file__))


def load_logreg_model_config(arch, n_classes, n_sources):
    config_path = join(_logreg_config_dir, f"{arch}.json")
    assert exists(config_path), f"no {arch}.json in {_logreg_config_dir}"
    config = load_json(config_path)
    config["n_classes"] = n_classes
    config["n_sources"] = n_sources
    config["vocab_size"] = VOCAB_SIZE
    return config


_LEXICON_USING_LEMMATIZE = {"liwc"}


def load_stock_lexicon_logreg_model_config(
    lexicon_name, arch, n_classes, n_sources, vocab_size
):
    config_path = join(_logreg_config_dir, f"{arch}.json")
    assert exists(config_path), f"no {arch}.json in {_logreg_config_dir}"
    config = load_json(config_path)
    config["n_classes"] = n_classes
    config["n_sources"] = n_sources
    config["vocab_size"] = vocab_size
    if lexicon_name in _LEXICON_USING_LEMMATIZE:
        config["use_lemmatize"] = True
    return config
