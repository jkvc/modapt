from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modapt.model.common import (
    MULTICLASS_STRATEGY,
    calc_multiclass_loss,
)
from modapt.model.model_utils import ReversalLayer
from modapt.model.zoo import register_model
from modapt.utils import AUTO_DEVICE


def elicit_lexicon(
    weights: np.ndarray, vocab: List[str], colnames: List[str]
) -> pd.DataFrame:
    nclass, vocabsize = weights.shape
    assert len(colnames) == nclass

    df = pd.DataFrame()
    df["word"] = vocab
    for c in range(nclass):
        df[colnames[c]] = weights[c]
    return df


@register_model
class LogisticRegressionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        self.multiclass_strategy = config["multiclass_strategy"]
        assert self.multiclass_strategy in MULTICLASS_STRATEGY

        self.vocab_size = config["vocab_size"]
        self.n_classes = config["n_classes"]
        self.n_sources = config["n_sources"]
        self.use_log_labelprop_bias = config["use_log_labelprop_bias"]
        self.use_learned_residualization = config["use_learned_residualization"]
        self.use_gradient_reversal = config["use_gradient_reversal"]
        self.hidden_size = config["hidden_size"]
        self.reg = config["reg"]

        self.tff = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.yout = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.cff = nn.Sequential(
            nn.Linear(self.n_sources, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.n_classes),
        )
        self.cout = nn.Sequential(
            ReversalLayer(),
            nn.Linear(self.hidden_size, self.n_sources),
        )

    def forward(self, batch):
        x = batch["x"].to(AUTO_DEVICE).to(torch.float)  # nsample, vocabsize
        nsample, vocabsize = x.shape
        assert vocabsize == self.vocab_size

        e = self.tff(x)
        logits = self.yout(e)

        if self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(AUTO_DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        if self.use_learned_residualization:
            if self.training:
                source_onehot = (
                    torch.eye(self.n_sources)[batch["source_idx"]]
                    .to(AUTO_DEVICE)
                    .to(torch.float)
                )
                clogits = self.cff(source_onehot)
                logits = clogits + logits

        labels = batch["y"].to(AUTO_DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

        if self.use_gradient_reversal:
            if self.training:
                confound_logits = self.cout(e)
                confound_loss, _ = calc_multiclass_loss(
                    confound_logits, batch["source_idx"].to(AUTO_DEVICE), "multinomial"
                )
                loss = loss + confound_loss

        # l1 reg on t weights only
        loss = loss + torch.abs(self.yout.weight @ self.tff.weight).sum() * self.reg
        loss = loss.mean()

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = (
            self.yout.weight.data.detach().cpu().numpy()
            @ self.tff.weight.data.detach().cpu().numpy()
        )
        return elicit_lexicon(weights, vocab, colnames)


@register_model
class LogisticRegressionSingularWeightMatrixModel(nn.Module):
    """
    Similar to `LogisticRegressionModel`, except the singular weight matrix
    This is useful if we want to populate the weight matrix from a stock lexicon
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.multiclass_strategy = config["multiclass_strategy"]
        assert self.multiclass_strategy in MULTICLASS_STRATEGY

        self.vocab_size = config["vocab_size"]
        self.n_classes = config["n_classes"]
        self.n_sources = config["n_sources"]

        self.use_adaptive_bias = config["use_adaptive_bias"]
        self.use_log_labelprop_bias = config["use_log_labelprop_bias"]
        assert not (self.use_adaptive_bias and self.use_log_labelprop_bias)

        # TODO maybe implement LR for fine-tuning stock lexicon weight
        # self.use_learned_residualization = config["use_learned_residualization"]
        self.reg = config["reg"]

        self.ff = nn.Linear(self.vocab_size, self.n_classes, bias=False)
        if self.use_adaptive_bias:
            self.bias = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, batch):
        x = batch["x"].to(AUTO_DEVICE).to(torch.float)  # nsample, vocabsize
        nsample, vocabsize = x.shape
        assert vocabsize == self.vocab_size

        logits = self.ff(x)

        if self.use_log_labelprop_bias:
            labelprops = (
                batch["labelprops"].to(AUTO_DEVICE).to(torch.float)
            )  # nsample, nclass
            logits = logits + torch.log(labelprops)

        if self.use_adaptive_bias:
            logits = logits + self.bias

        labels = batch["y"].to(AUTO_DEVICE)
        loss, labels = calc_multiclass_loss(logits, labels, self.multiclass_strategy)

        # l1 reg on t weights only TODO enable
        # loss = loss + torch.abs(self.ff.weight).sum() * self.reg
        loss = loss + torch.abs(self.bias).sum() * self.reg
        loss = loss.mean()

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
        }

    def set_weight_from_lexicon(self, df: pd.DataFrame, colnames: List[str]):
        weight_shape = self.ff.weight.data.detach().cpu().numpy().shape
        nclass, vocabsize = weight_shape
        assert len(colnames) == nclass

        weights = np.zeros(weight_shape)
        for c in range(nclass):
            weights[c] = df[colnames[c]]

        with torch.no_grad():
            self.ff.weight.copy_(torch.FloatTensor(weights))
            # freeze, dont fine tune stock lexicon
            # TODO maybe implement to enable finetune for LR
            self.ff.weight.requires_grad = False
