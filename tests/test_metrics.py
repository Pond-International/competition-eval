"""
Unit tests for metrics module.
"""
import numpy as np
import pytest
import pandas as pd
from metrics import (
    accuracy,
    root_mean_squared_error,
    weighted_mean_squared_error,
    precision_at_k,
    discounted_cumulative_gain,
    auc,
)

# Common test data
@pytest.fixture
def valid_binary_data():
    return np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])

@pytest.fixture
def valid_continuous_data():
    return np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 2.1, 3.1, 4.1])

@pytest.fixture
def valid_rec_data():
    y_true = pd.DataFrame({
        "ADDRESS": ["addr1", "addr1", "addr2", "addr2"],
        "REC": ["rec1", "rec2", "rec3", "rec4"]
    })
    y_pred = pd.DataFrame({
        "ADDRESS": ["addr1", "addr1", "addr2", "addr2"],
        "REC": ["rec1", "rec3", "rec3", "rec4"]
    })
    return y_true, y_pred

# Test accuracy metric
def test_accuracy_valid(valid_binary_data):
    y_true, y_pred = valid_binary_data
    assert accuracy(y_true, y_pred) == 0.75

def test_accuracy_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        accuracy(np.array([]), np.array([]))

def test_accuracy_nan():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        accuracy(np.array([1, np.nan]), np.array([1, 0]))

def test_accuracy_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        accuracy(np.array([1, "a"]), np.array([1, 0]))

def test_accuracy_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        accuracy(np.array([1, 0]), np.array([1]))

def test_accuracy_not_numpy():
    with pytest.raises(ValueError, match="Inputs must be numpy arrays"):
        accuracy([1, 0], [1, 0])

# Test mean squared error metric
def test_mse_valid(valid_continuous_data):
    y_true, y_pred = valid_continuous_data
    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(0.01))

def test_mse_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        root_mean_squared_error(np.array([]), np.array([]))

def test_mse_nan():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        root_mean_squared_error(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

def test_mse_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        root_mean_squared_error(np.array([1, "a"]), np.array([1, 2]))

def test_mse_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        root_mean_squared_error(np.array([1.0, 2.0]), np.array([1.0]))

def test_mse_not_numpy():
    with pytest.raises(ValueError, match="Inputs must be numpy arrays"):
        root_mean_squared_error([1.0, 2.0], [1.0, 2.0])

# Test weighted mean squared error metric
def test_wmse_valid(valid_continuous_data):
    y_true, y_pred = valid_continuous_data
    expected = np.sqrt(np.mean(np.abs(y_true) * (y_true - y_pred) ** 2))
    assert np.isclose(weighted_mean_squared_error(y_true, y_pred), expected)

def test_wmse_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        weighted_mean_squared_error(np.array([]), np.array([]))

def test_wmse_nan_values():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        weighted_mean_squared_error(
            np.array([1.0, np.nan]), 
            np.array([1.0, 2.0]), 
        )

def test_wmse_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        weighted_mean_squared_error(
            np.array([1, "a"]), 
            np.array([1, 2]), 
        )

def test_wmse_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        weighted_mean_squared_error(
            np.array([1.0, 2.0]), 
            np.array([1.0]), 
        )

def test_wmse_not_numpy():
    with pytest.raises(ValueError, match="Inputs must be numpy arrays"):
        weighted_mean_squared_error([1.0, 2.0], [1.0, 2.0])

# Test precision at k metric
def test_precision_at_k_valid(valid_rec_data):
    y_true, y_pred = valid_rec_data
    # For addr1: 1/2 matches out of 2 recs
    # For addr2: 2/2 matches out of 2 recs
    # Average precision = (0.5 + 1.0)/2 = 0.75
    assert np.isclose(precision_at_k(y_true, y_pred, k=2), 0.75)

def test_precision_at_k_empty():
    y_true = pd.DataFrame(columns=["ADDRESS", "REC"])
    y_pred = pd.DataFrame(columns=["ADDRESS", "REC"])
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        precision_at_k(y_true, y_pred, k=1)

def test_precision_at_k_missing_addresses():
    y_true = pd.DataFrame({
        "ADDRESS": ["addr1", "addr2"],
        "REC": ["rec1", "rec2"]
    })
    y_pred = pd.DataFrame({
        "ADDRESS": ["addr1"],
        "REC": ["rec1"]
    })
    with pytest.raises(ValueError, match="Missing recommendations for some addresses"):
        precision_at_k(y_true, y_pred, k=1)

def test_precision_at_k_invalid_input_type():
    with pytest.raises(ValueError, match="Inputs must be pandas DataFrames"):
        precision_at_k(
            np.array([1, 2]),
            np.array([1, 2]),
            k=1
        )

def test_precision_at_k_perfect_match(valid_rec_data):
    y_true, _ = valid_rec_data
    # Perfect match case
    y_pred = y_true.copy()
    assert np.isclose(precision_at_k(y_true, y_pred, k=2), 1.0)

def test_precision_at_k_no_match(valid_rec_data):
    y_true, _ = valid_rec_data
    # No matches case
    y_pred = pd.DataFrame({
        "ADDRESS": ["addr1", "addr1", "addr2", "addr2"],
        "REC": ["rec5", "rec6", "rec7", "rec8"]
    })
    assert np.isclose(precision_at_k(y_true, y_pred, k=2), 0.0)

# Test discounted cumulative gain metric
def test_dcg_valid():
    true_relevance = np.array([3.0, 2.0, 1.0])
    ranks = np.array([1, 2, 3])
    expected = (3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4))/3
    assert np.isclose(discounted_cumulative_gain(true_relevance, ranks), expected)

def test_dcg_single_item():
    true_relevance = np.array([1.0])
    ranks = np.array([1])
    expected = 1.0 / np.log2(2)
    assert np.isclose(discounted_cumulative_gain(true_relevance, ranks), expected)

def test_dcg_zero_relevance():
    true_relevance = np.array([0.0, 0.0, 0.0])
    ranks = np.array([1, 2, 3])
    assert discounted_cumulative_gain(true_relevance, ranks) == 0.0

def test_dcg_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        discounted_cumulative_gain(np.array([]), np.array([]))

def test_dcg_nan():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        discounted_cumulative_gain(np.array([1, np.nan]), np.array([1, 2]))

def test_dcg_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        discounted_cumulative_gain(np.array([1, "a"]), np.array([1, 2]))

def test_dcg_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        discounted_cumulative_gain(np.array([1, 2]), np.array([1]))

def test_dcg_non_positive_ranks():
    with pytest.raises(ValueError, match="Ranks must be positive integers"):
        discounted_cumulative_gain(np.array([1.0, 2.0]), np.array([0, 1]))

def test_dcg_duplicate_ranks():
    with pytest.raises(ValueError, match="Ranks must not contain any duplicates"):
        discounted_cumulative_gain(np.array([1.0, 2.0]), np.array([1, 1]))

def test_dcg_non_integer_ranks():
    with pytest.raises(ValueError, match="Ranks must be positive integers"):
        discounted_cumulative_gain(np.array([1.0, 2.0]), np.array([1.5, 2.5]))

# Test AUC metric
def test_auc_valid():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(auc(y_true, y_pred), 0.75)

def test_auc_perfect_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.1, 0.9])
    assert np.isclose(auc(y_true, y_pred), 1.0)

def test_auc_random_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(auc(y_true, y_pred), 0.5)

def test_auc_empty():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        auc(np.array([]), np.array([]))

def test_auc_nan():
    with pytest.raises(ValueError, match="Predictions cannot contain missing values"):
        auc(np.array([1, np.nan]), np.array([0.5, 0.8]))

def test_auc_non_numeric():
    with pytest.raises(ValueError, match="Ground truth and predictions must contain only numeric values"):
        auc(np.array([1, "a"]), np.array([0.5, 0.8]))

def test_auc_different_lengths():
    with pytest.raises(ValueError, match="Ground truth and predictions must have the same length"):
        auc(np.array([0, 1]), np.array([0.5]))

def test_auc_not_numpy():
    with pytest.raises(ValueError, match="Inputs must be numpy arrays"):
        auc([0, 1], [0.5, 0.8])
