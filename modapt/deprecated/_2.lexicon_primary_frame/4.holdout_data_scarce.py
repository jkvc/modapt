import sys
from os.path import join
from random import Random

from config import ISSUES, LEX_DIR
from modapt.eval import reduce_and_save_metrics
from modapt.learning import _print_metrics
from modapt.lexicon import eval_lexicon_model, run_lexicon_experiment
from modapt.text_samples import load_all_text_samples
from modapt.utils import save_json

_arch = sys.argv[1]

# WEIGHT_DECAYS = [2, 3, 4, 5]
WEIGHT_DECAYS = [1]

NUM_TRAIN_SAMPLE = 500

RNG = Random()
RNG.seed(0xDEADBEEF)

if __name__ == "__main__":
    for weight_decay in WEIGHT_DECAYS:
        for holdout_issue in ISSUES:
            print(">> holdout", holdout_issue)
            logdir = join(
                LEX_DIR, f"4.{_arch}", str(weight_decay), f"holdout_{holdout_issue}"
            )

            train_issues = [i for i in ISSUES if i != holdout_issue]
            train_samples = load_all_text_samples(
                train_issues, "train", "primary_frame"
            )
            RNG.shuffle(train_samples)
            train_samples = train_samples[:NUM_TRAIN_SAMPLE]

            vocab, model, train_metrics = run_lexicon_experiment(
                _arch,
                train_samples,
                logdir,
                weight_decay=weight_decay,
            )

            valid_samples = load_all_text_samples(
                [holdout_issue], "train", "primary_frame"
            )
            valid_metrics = eval_lexicon_model(model, valid_samples, vocab)

            leaf_metrics = {}
            for prefix, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
                for k, v in metrics.items():
                    leaf_metrics[f"{prefix}_{k}"] = v

            _print_metrics(leaf_metrics)
            save_json(
                leaf_metrics,
                join(logdir, "leaf_metrics.json"),
            )

    reduce_and_save_metrics(join(LEX_DIR, f"4.{_arch}"))
