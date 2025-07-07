"""
Unit tests for area under the precision–recall curve metric.
"""
import numpy as np
import pytest
from metrics import auc_pr

def test_auc_pr_valid():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(auc_pr(y_true, y_pred), 0.8333333333333333)

def test_auc_pr_perfect_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.1, 0.9])
    assert np.isclose(auc_pr(y_true, y_pred), 1.0)

def test_auc_pr_random_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(auc_pr(y_true, y_pred), 0.5)

def test_auc_pr_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        auc_pr(np.array([]), np.array([]))

def test_auc_pr_nan():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        auc_pr(np.array([1, np.nan]), np.array([1, 0]))

def test_auc_pr_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        auc_pr(np.array([1, "a"]), np.array([1, 0]))

def test_auc_pr_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        auc_pr(np.array([1, 0]), np.array([1]))

def test_auc_pr_not_numpy():
    with pytest.raises(ValueError, match="Inputs must be numpy arrays"):
        auc_pr([1, 0], [1, 0])
