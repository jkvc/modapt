from os.path import dirname, join, realpath

import pandas as pd
from modapt.modapt import logreg_predict_eval, logreg_train

_THIS_DIR = dirname(realpath(__file__))

print(">> train logreg model")
model, vocab, metrics = logreg_train(
    data=join(_THIS_DIR, "demo_data", "train.csv"),
    save_model_dir=join(_THIS_DIR, "models", "logreg"),
    vocab_size=100,
)
print(">> train logreg model done")

print(">> predict eval logreg model")
scores, est_metrics = logreg_predict_eval(
    data_labeled=join(_THIS_DIR, "demo_data", "valid_labeled.csv"),
    data_unlabeled=join(_THIS_DIR, "demo_data", "valid_unlabeled.csv"),
    model_dir=join(_THIS_DIR, "models", "logreg"),
)
print(">> predict eval logreg model done")
print(">> estimated acc:", est_metrics["mean_estimated_acc"])
print(">> scores shape:", scores.shape)
