"""
Metrics module for evaluating predictions against ground truth.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def _validate_numeric_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
        
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input data cannot be empty")
        
    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and predictions must have the same length")
        
    # Check for non-numeric values first
    if not np.issubdtype(y_true.dtype, np.number) or not np.issubdtype(y_pred.dtype, np.number):
        raise ValueError("Ground truth and predictions must contain only numeric values")
        
    # Only check for NaN if arrays are numeric
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Predictions cannot contain missing values")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy between ground truth and predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error between ground truth and predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: Mean squared error
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error between ground truth and predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: Mean squared error
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)

def weighted_mean_squared_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
) -> float:
    """Calculate weighted mean squared error between ground truth and predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: Weighted mean squared error
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(y_true, y_pred)
    return np.sqrt(np.mean(np.abs(y_true) * (y_true - y_pred)**2))

def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Area Under the ROC Curve (AUC) between ground truth and predictions.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        
    Returns:
        float: AUC score
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(y_true, y_pred)
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("Predictions must be between 0 and 1")
    return float(roc_auc_score(y_true, y_pred))

def _validate_rec_inputs(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise ValueError("Inputs must be pandas DataFrames")
        
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input data cannot be empty")
        
    if len(set(y_true["ADDRESS"]) - set(y_pred["ADDRESS"])) > 0:
        raise ValueError("Missing recommendations for some addresses")
        
def precision_at_k(
    y_true: pd.DataFrame, 
    y_pred: pd.DataFrame, 
    k: int
) -> float:
    """Calculate precision at k between ground truth and recommendations.
    
    Args:
        y_true: Ground truth items
        y_pred: Recommendations
        k: Number of items recommended
        
    Returns:
        float: Precision at k score
        
    Raises:
        ValueError: If inputs are invalid or k is not a positive integer
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
        
    _validate_rec_inputs(y_true, y_pred)
    
    y_pred["score"] = 1
    combined_df = y_true.merge(y_pred, on=["ADDRESS", "REC"], how="left")
    combined_df = combined_df.groupby("ADDRESS").sum("score")
    precision = combined_df/k

    return float(precision.mean().item())

def discounted_cumulative_gain(
    true_relavance: np.ndarray, 
    ranks: np.ndarray
) -> float:
    """Calculate normalized discounted cumulative gain between ground truth relavance score and ranks.
    
    Args:
        true_relavance: Ground truth relavance score
        ranks: Recommendations rank
        
    Returns:
        float: Normalized discounted cumulative gain score
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_numeric_inputs(true_relavance, ranks)
    if not np.issubdtype(ranks.dtype, np.integer) or np.any(ranks <= 0):
        raise ValueError("Ranks must be positive integers")
    if len(ranks) != len(set(ranks)):
        raise ValueError("Ranks must not contain any duplicates")
    discount = 1 / np.log2(ranks + 1)
    dcg = np.mean(true_relavance * discount)
    return dcg

def pairwise_cost(
    y_true: pd.DataFrame, 
    y_pred: pd.DataFrame
) -> float:
    """Calculate pairwise cost between ground truth and recommendations.
    
    Args:
        y_true: Ground truth items
        y_pred: Recommendations
        
    Returns:
        float: Pairwise cost score
        
    Raises:
        TypeError: If inputs are not pandas DataFrames
        ValueError: If inputs are invalid or missing required data
    """
    if not isinstance(y_true, pd.DataFrame):
        raise TypeError("y_true must be a pandas DataFrame")
    if not isinstance(y_pred, pd.DataFrame):
        raise TypeError("y_pred must be a pandas DataFrame")
    
    # Check for empty DataFrames
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Missing weights in submission data")
        
    # Cost on pairs where TARGET != "originality"
    y_true_pair = y_true[y_true["TARGET"]!="originality"]
    combined_df = y_true_pair.merge(
        y_pred, 
        left_on=["SOURCE_A", "TARGET"], 
        right_on=["SOURCE", "TARGET"],
        how="left")
    if combined_df['WEIGHT'].isnull().any():
        raise ValueError("Missing weights in submission data")
    combined_df.rename(columns={"WEIGHT": "WEIGHT_A"}, inplace=True)
    
    combined_df = combined_df.merge(
        y_pred, 
        left_on=["SOURCE_B", "TARGET"], 
        right_on=["SOURCE", "TARGET"],
        how="left")
    if combined_df['WEIGHT'].isnull().any():
        raise ValueError("Missing weights in submission data")
    combined_df.rename(columns={"WEIGHT": "WEIGHT_B"}, inplace=True)
    
    cost_pair = ((np.log(combined_df["WEIGHT_B"]) - np.log(combined_df["WEIGHT_A"]) - combined_df["B_OVER_A"])**2).sum()

    # Cost on pairs where TARGET == "originality"
    y_true_origin = y_true[y_true["TARGET"]=="originality"]
    if len(y_true_origin) == 0:
        cost_origin = 0
    else:
        combined_df = y_true_origin.merge(
            y_pred, 
            left_on=["SOURCE_A", "TARGET"], 
            right_on=["SOURCE", "TARGET"],
            how="left")
        if combined_df['WEIGHT'].isnull().any():
            raise ValueError("Missing weights in submission data")
        cost_origin = ((np.log(combined_df["WEIGHT"]) - combined_df["B_OVER_A"])**2).sum()

    return float(cost_pair) + float(cost_origin)

# Dictionary mapping metric names to functions
METRICS = {
    'accuracy': accuracy,
    'rmse': root_mean_squared_error,
    'wmse': weighted_mean_squared_error,
    'precision_at_k': precision_at_k,
    'dcg': discounted_cumulative_gain,
    'auc': auc,
    'mse': mean_squared_error,
    'pairwise_cost': pairwise_cost
}
