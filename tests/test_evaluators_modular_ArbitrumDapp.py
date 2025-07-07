

import pandas as pd
import pytest
import numpy as np

from src.evaluators import ModularEvaluator


@pytest.fixture
def sample_gt_sub():
    # Ground-truth and submission both include HISTORICAL_CATEGORY
    gt = pd.DataFrame({
        "ADDRESS":              ["a", "b", "c", "d"],
        "DAPP_ADDRESS":         ["x", "x", "y", "y"],
        "HISTORICAL_CATEGORY":  ["first_time_tx", "first_time_tx", "repeat_tx", "repeat_tx"],
        "LABEL":                [0, 1, 0, 1],
    })
    sub = pd.DataFrame({
        "ADDRESS":              ["a", "b", "c", "d"],
        "DAPP_ADDRESS":         ["x", "x", "y", "y"],
        "HISTORICAL_CATEGORY":  ["first_time_tx", "first_time_tx", "repeat_tx", "repeat_tx"],
        "LABEL":                [0.1, 0.9, 0.2, 0.8],
    })
    return gt, sub


@pytest.fixture
def arb_config():
    # match configs/arbitrum_dapp.yaml
    return {
        "LABEL": {
            "metric": "auc",
            "weight": 1.0,
            "group_by": "HISTORICAL_CATEGORY",
            "group_weights": {
                "first_time_tx": 0.6, 
                "repeat_tx": 0.4},
        }
    }


def test_modular_arbitrumdapp_happy_path(sample_gt_sub, arb_config):
    gt, sub = sample_gt_sub
    evalr = ModularEvaluator()
    evalr.config = arb_config

    score = evalr.compute_metric(gt, sub)
    # Each group perfect separation => AUC=1.0, weighted sum = 0.6*1 + 0.4*1 = 1.0
    assert pytest.approx(score) == 1.0


def test_modular_arbitrumdapp_inverse_flag(sample_gt_sub, arb_config):
    gt, sub = sample_gt_sub
    # flip predictions so each group AUC=0
    sub2 = sub.copy()
    sub2["LABEL"] = 1 - sub2["LABEL"]

    cfg = arb_config.copy()
    cfg["LABEL"] = cfg["LABEL"].copy()
    cfg["LABEL"]["inverse"] = True

    evalr = ModularEvaluator()
    evalr.config = cfg

    score = evalr.compute_metric(gt, sub2)
    # without inverse , get 0; with inverse => 1
    assert pytest.approx(score) == 1.0



def test_modular_arbitrumdapp_missing_group(sample_gt_sub):
    gt, sub = sample_gt_sub
    bad_cfg = {
        "LABEL": {
            "metric": "auc",
            "weight": 1.0,
            "group_by": "HISTORICAL_CATEGORY",
            "group_weights": { "first_time_tx" : 0.5,
                               "bogus_group": 0.5},
        }
    }
    evalr = ModularEvaluator()
    evalr.config = bad_cfg

    with pytest.raises(ValueError, match="Missing group"):
        evalr.compute_metric(gt, sub)
