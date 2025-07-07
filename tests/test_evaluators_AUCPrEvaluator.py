"""
Unit tests for AUCPrEvaluator class.
"""
import numpy as np
import pytest
from unittest.mock import patch

from src.evaluators import AUCPrEvaluator


@pytest.fixture
def auc_pr_evaluator():
    """Create an AUC-PR evaluator instance."""
    return AUCPrEvaluator()


def test_auc_pr_evaluator_init(auc_pr_evaluator):
    """Test initialization of AUCPrEvaluator."""
    assert auc_pr_evaluator.name == "auc_pr"
    assert auc_pr_evaluator.description == "Area Under the Precision–Recall Curve"


@patch("metrics.auc_pr")
def test_auc_pr_evaluator_compute_metric(mock_auc_pr, auc_pr_evaluator):
    """Test compute_metric method of AUCPrEvaluator."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.3, 0.9])

    mock_auc_pr.return_value = 0.85

    result = auc_pr_evaluator.compute_metric(y_true, y_pred)

    mock_auc_pr.assert_called_once_with(y_true, y_pred)
    assert result == 0.85


def test_auc_pr_evaluator_get_default_params(auc_pr_evaluator):
    """Test get_default_params returns an empty dict."""
    params = auc_pr_evaluator.get_default_params()
    assert isinstance(params, dict)
    assert params == {}
