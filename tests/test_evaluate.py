"""
Unit tests for evaluate module.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from evaluate import (
    load_data,
    process_supervised_data,
    process_recommend_data,
    compute_metric
)

# Test fixtures
@pytest.fixture
def sample_numeric_data():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3"],
        "LABEL": [1.0, 2.0, 3.0]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3"],
        "PRED": [1.1, 2.1, 3.1]
    })
    return ground_truth, submission

@pytest.fixture
def sample_rec_data():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1", "ADDR2", "ADDR2"],
        "REC": ["REC1", "REC2", "REC3", "REC4"]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1", "ADDR2", "ADDR2"],
        "REC": ["REC1", "REC3", "REC3", "REC4"]
    })
    return ground_truth, submission

@pytest.fixture
def temp_data_files():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.1, 2.1]
    })

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        ground_truth.to_parquet(gt_file.name)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        submission.to_csv(sub_file.name, index=False)

    yield gt_file.name, sub_file.name

    os.unlink(gt_file.name)
    os.unlink(sub_file.name)

@pytest.fixture
def sample_split_data():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3", "ADDR4"],
        "LABEL": [1.0, 2.0, 3.0, 4.0],
        "SPLIT": ["public", "public", "private", "private"]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3", "ADDR4"],
        "PRED": [1.1, 2.1, 3.1, 4.1]
    })
    return ground_truth, submission

@pytest.fixture
def sample_split_rec_data():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1", "ADDR2", "ADDR2", "ADDR3", "ADDR3"],
        "REC": ["REC1", "REC2", "REC3", "REC4", "REC5", "REC6"],
        "SPLIT": ["public", "public", "public", "public", "private", "private"]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1", "ADDR2", "ADDR2", "ADDR3", "ADDR3"],
        "REC": ["REC1", "REC3", "REC3", "REC4", "REC5", "REC6"]
    })
    return ground_truth, submission

@pytest.fixture
def temp_split_data_files():
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3", "ADDR4"],
        "LABEL": [1.0, 2.0, 3.0, 4.0],
        "SPLIT": ["public", "public", "private", "private"]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2", "ADDR3", "ADDR4"],
        "LABEL": [1.1, 2.1, 3.1, 4.1]
    })

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        ground_truth.to_parquet(gt_file.name)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        submission.to_csv(sub_file.name, index=False)

    yield gt_file.name, sub_file.name

    os.unlink(gt_file.name)
    os.unlink(sub_file.name)

# Test load_data
def test_load_data(temp_data_files):
    gt_path, sub_path = temp_data_files
    gt_df, sub_df = load_data(gt_path, sub_path)
    
    assert isinstance(gt_df, pd.DataFrame)
    assert isinstance(sub_df, pd.DataFrame)
    assert gt_df.shape == (2, 2)
    assert sub_df.shape == (2, 2)

def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.parquet", "nonexistent.csv")

def test_load_data_with_split(temp_split_data_files):
    gt_path, sub_path = temp_split_data_files
    gt_df, sub_df = load_data(gt_path, sub_path, custom_split="public")
    
    assert isinstance(gt_df, pd.DataFrame)
    assert isinstance(sub_df, pd.DataFrame)
    assert gt_df.shape == (4, 3)  # Should include the split column
    assert sub_df.shape == (4, 2)  # Should not include split column

def test_load_data_missing_split_column():
    # Create data without split column
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0]
    })
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.1, 2.1]
    })

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        ground_truth.to_parquet(gt_file.name)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        submission.to_csv(sub_file.name, index=False)

    with pytest.raises(ValueError, match="Ground truth data does not contain a 'split' column"):
        load_data(gt_file.name, sub_file.name, custom_split="public")

    os.unlink(gt_file.name)
    os.unlink(sub_file.name)

# Test validate_numeric_data
def test_validate_numeric_data_valid(sample_numeric_data):
    gt_df, sub_df = sample_numeric_data
    result = process_supervised_data(gt_df, sub_df, data_portion=1.0, after_split=False)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"ADDRESS", "LABEL", "PRED"}
    assert len(result) == 3

def test_validate_numeric_data_portion(sample_numeric_data):
    gt_df, sub_df = sample_numeric_data
    # Test with first half
    result = process_supervised_data(gt_df, sub_df, data_portion=0.5, after_split=False)
    assert len(result) == 1  # Since we have 3 rows, 0.5 rounds down to 1
    assert result["ADDRESS"].iloc[0].upper() == "ADDR1"
    
    # Test with remaining half
    result = process_supervised_data(gt_df, sub_df, data_portion=0.5, after_split=True)
    assert len(result) == 2
    assert result["ADDRESS"].iloc[0].upper() == "ADDR2"
    assert result["ADDRESS"].iloc[1].upper() == "ADDR3"

def test_validate_numeric_data_invalid_portion(sample_numeric_data):
    gt_df, sub_df = sample_numeric_data
    # Test portion > 1
    with pytest.raises(ValueError, match="data-portion must be between 0 and 1"):
        process_supervised_data(gt_df, sub_df, data_portion=1.5, after_split=False)
    
    # Test portion <= 0
    with pytest.raises(ValueError, match="data-portion must be between 0 and 1"):
        process_supervised_data(gt_df, sub_df, data_portion=0, after_split=False)
    
    # Test portion = 1 with after_split
    with pytest.raises(ValueError, match="data-portion must be less than 1 when after_split is enabled"):
        process_supervised_data(gt_df, sub_df, data_portion=1.0, after_split=True)

def test_validate_numeric_data_missing_address():
    gt_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0]
    })
    sub_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR3"],
        "PRED": [1.1, 3.1]
    })
    result = process_supervised_data(gt_df, sub_df)
    assert pd.isna(result.loc[result["ADDRESS"] == "ADDR2", "PRED"]).all()

def test_validate_numeric_data_duplicate_address():
    gt_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0]
    })
    sub_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1"],
        "PRED": [1.1, 1.2]
    })
    with pytest.raises(ValueError, match="Duplicate addresses found in submission data"):
        process_supervised_data(gt_df, sub_df)

def test_validate_numeric_data_case_insensitive():
    gt_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0]
    })
    sub_df = pd.DataFrame({
        "ADDRESS": ["addr1", "addr2"],
        "PRED": [1.1, 2.1]
    })
    result = process_supervised_data(gt_df, sub_df)
    assert len(result) == 2
    assert not pd.isna(result["PRED"]).any()

def test_process_supervised_data_with_integer_ids():
    """Test that process_supervised_data handles integer IDs correctly."""
    # Create test data with integer IDs
    ground_truth_df = pd.DataFrame({
        0: [123, 456, 789, 101112],  # Integer IDs
        1: [1, 0, 1, 0]
    })
    
    submission_df = pd.DataFrame({
        "ADDRESS": [456, 789, 123, 101112],  # Same integers but different order
        "PRED": [0.2, 0.8, 0.6, 0.1]
    })
    
    # Process the data
    result_df = process_supervised_data(ground_truth_df, submission_df)
    
    # Check that IDs were converted to strings
    assert result_df["ADDRESS"].dtype == object  # string type
    assert all(isinstance(id_val, str) for id_val in result_df["ADDRESS"])
    
    # Check that the data was properly aligned
    expected_order = ["123", "456", "789", "101112"]
    assert list(result_df["ADDRESS"]) == expected_order
    assert list(result_df["LABEL"]) == [1, 0, 1, 0]
    assert list(result_df["PRED"]) == [0.6, 0.2, 0.8, 0.1]
    
    # Test with mixed string and integer IDs
    ground_truth_df = pd.DataFrame({
        0: [123, "ABC", 789, "DEF"],  # Mixed IDs
        1: [1, 0, 1, 0]
    })
    
    submission_df = pd.DataFrame({
        "ADDRESS": ["ABC", 789, 123, "DEF"],  # Mixed IDs in different order
        "PRED": [0.2, 0.8, 0.6, 0.1]
    })
    
    # Process the data
    result_df = process_supervised_data(ground_truth_df, submission_df)
    
    # Check that IDs were converted to strings and uppercased where applicable
    assert result_df["ADDRESS"].dtype == object
    assert all(isinstance(id_val, str) for id_val in result_df["ADDRESS"])
    assert all(id_val.isupper() for id_val in result_df["ADDRESS"] if not id_val.isdigit())
    
    # Check that the data was properly aligned
    expected_order = ["123", "ABC", "789", "DEF"]
    assert list(result_df["ADDRESS"]) == expected_order
    assert list(result_df["LABEL"]) == [1, 0, 1, 0]
    assert list(result_df["PRED"]) == [0.6, 0.2, 0.8, 0.1]

def test_process_supervised_data_with_split(sample_split_data):
    gt_df, sub_df = sample_split_data
    
    # Test public split
    result = process_supervised_data(gt_df, sub_df, custom_split="public")
    assert len(result) == 2
    assert set(result["ADDRESS"]) == {"ADDR1", "ADDR2"}
    
    # Test private split
    result = process_supervised_data(gt_df, sub_df, custom_split="private")
    assert len(result) == 2
    assert set(result["ADDRESS"]) == {"ADDR3", "ADDR4"}

def test_process_supervised_data_split_case_insensitive(sample_split_data):
    gt_df, sub_df = sample_split_data
    gt_df["SPLIT"] = gt_df["SPLIT"].str.upper()  # Make split values uppercase
    
    result = process_supervised_data(gt_df, sub_df, custom_split="public")
    assert len(result) == 2
    assert set(result["ADDRESS"]) == {"ADDR1", "ADDR2"}

def test_process_recommend_data_with_split(sample_split_rec_data):
    gt_df, sub_df = sample_split_rec_data
    
    # Test public split
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, custom_split="public")
    assert len(result_gt) == 4  # Two addresses with two recommendations each
    assert set(result_gt["ADDRESS"].unique()) == {"ADDR1", "ADDR2"}
    
    # Test private split
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, custom_split="private")
    assert len(result_gt) == 2  # One address with two recommendations
    assert set(result_gt["ADDRESS"].unique()) == {"ADDR3"}

def test_process_recommend_data_split_case_insensitive(sample_split_rec_data):
    gt_df, sub_df = sample_split_rec_data
    gt_df["SPLIT"] = gt_df["SPLIT"].str.upper()  # Make split values uppercase
    
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, custom_split="public")
    assert len(result_gt) == 4
    assert set(result_gt["ADDRESS"].unique()) == {"ADDR1", "ADDR2"}

def test_invalid_custom_split_value(sample_split_data):
    gt_df, sub_df = sample_split_data
    
    with pytest.raises(ValueError, match="custom_split must be either 'public' or 'private'"):
        process_supervised_data(gt_df, sub_df, custom_split="invalid")
    
    with pytest.raises(ValueError, match="custom_split must be either 'public' or 'private'"):
        gt_df_rec = pd.DataFrame({
            "ADDRESS": ["ADDR1", "ADDR2"],
            "REC": ["REC1", "REC2"],
            "SPLIT": ["public", "private"]
        })
        sub_df_rec = pd.DataFrame({
            "ADDRESS": ["ADDR1", "ADDR2"],
            "REC": ["REC1", "REC2"]
        })
        process_recommend_data(gt_df_rec, sub_df_rec, custom_split="invalid")

def test_data_portion_with_custom_split(sample_split_data):
    gt_df, sub_df = sample_split_data
    
    # Test through main function to catch the argument validation
    import sys
    from io import StringIO
    from evaluate import main
    
    # Capture stderr to check error message
    stderr = StringIO()
    sys.stderr = stderr
    
    # Create temporary files for the test
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        gt_df.to_parquet(gt_file.name)
        
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        sub_df.to_csv(sub_file.name, index=False)
    
    # Test with both data-portion and custom-split
    sys.argv = [
        "evaluate.py",
        gt_file.name,
        sub_file.name,
        "rmse",
        "--data-portion", "0.5",
        "--custom-split", "public"
    ]
    
    with pytest.raises(SystemExit):
        main()
    
    error_output = stderr.getvalue()
    assert "--data-portion and --custom-split cannot be used together" in error_output
    
    # Cleanup
    os.unlink(gt_file.name)
    os.unlink(sub_file.name)
    sys.stderr = sys.__stderr__

# Test validate_rec_data
def test_validate_rec_data_valid(sample_rec_data):
    gt_df, sub_df = sample_rec_data
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, data_portion=1.0, after_split=False)
    
    assert isinstance(result_gt, pd.DataFrame)
    assert isinstance(result_sub, pd.DataFrame)
    assert len(result_gt) == len(gt_df)
    assert len(result_sub) == len(sub_df)

def test_validate_rec_data_portion(sample_rec_data):
    gt_df, sub_df = sample_rec_data
    # Test with first half of unique addresses
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, data_portion=0.5, after_split=False)
    assert len(set(result_gt["ADDRESS"])) == 1  # Only ADDR1
    assert "ADDR1" in result_gt["ADDRESS"].values
    assert "ADDR2" not in result_gt["ADDRESS"].values
    
    # Test with remaining half
    result_gt, result_sub = process_recommend_data(gt_df, sub_df, data_portion=0.5, after_split=True)
    assert len(set(result_gt["ADDRESS"])) == 1  # Only ADDR2
    assert "ADDR1" not in result_gt["ADDRESS"].values
    assert "ADDR2" in result_gt["ADDRESS"].values

def test_validate_rec_data_invalid_portion(sample_rec_data):
    gt_df, sub_df = sample_rec_data
    # Test portion > 1
    with pytest.raises(ValueError, match="data-portion must be between 0 and 1"):
        process_recommend_data(gt_df, sub_df, data_portion=1.5, after_split=False)
    
    # Test portion <= 0
    with pytest.raises(ValueError, match="data-portion must be between 0 and 1"):
        process_recommend_data(gt_df, sub_df, data_portion=0, after_split=False)
    
    # Test portion = 1 with after_split
    with pytest.raises(ValueError, match="data-portion must be less than 1 when after_split is enabled"):
        process_recommend_data(gt_df, sub_df, data_portion=1.0, after_split=True)

def test_validate_rec_data_case_insensitive():
    gt_df = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1"],
        "REC": ["REC1", "REC2"]
    })
    sub_df = pd.DataFrame({
        "ADDRESS": ["addr1", "addr1"],
        "REC": ["rec1", "rec3"]
    })
    result_gt, result_sub = process_recommend_data(gt_df, sub_df)
    assert (result_gt["ADDRESS"] == result_gt["ADDRESS"].str.upper()).all()
    assert (result_gt["REC"] == result_gt["REC"].str.upper()).all()
    assert (result_sub["ADDRESS"] == result_sub["ADDRESS"].str.upper()).all()
    assert (result_sub["REC"] == result_sub["REC"].str.upper()).all()

# Test compute_metric
def test_compute_metric_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    score = compute_metric(y_true, y_pred, "rmse")
    assert np.isclose(score, np.sqrt(0.01))

def test_compute_metric_precision():
    y_true = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1"],
        "REC": ["REC1", "REC2"]
    })
    y_pred = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR1"],
        "REC": ["REC1", "REC3"]
    })
    score = compute_metric(y_true, y_pred, "precision_at_k", {"k": 2})
    assert score == 0.5

def test_compute_metric_invalid():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 2.1])
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metric(y_true, y_pred, "invalid_metric")
