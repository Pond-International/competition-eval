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
    compute_metric,
    process_pairwise_data,
    load_process_data_deepfunding
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

def test_load_data_skip_column_check():
    # Create ground truth with extra column
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0],
        "EXTRA": ["X", "Y"]
    })
    # Create submission with different columns
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.1, 2.1]
    })

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        ground_truth.to_parquet(gt_file.name)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        submission.to_csv(sub_file.name, index=False)

    # Should fail without skip_column_check
    with pytest.raises(ValueError, match="Unexpected column names in submission"):
        load_data(gt_file.name, sub_file.name)

    # Should succeed with skip_column_check
    gt_df, sub_df = load_data(gt_file.name, sub_file.name, skip_column_check=True)
    assert set(gt_df.columns) == {"ADDRESS", "LABEL", "EXTRA"}
    assert set(sub_df.columns) == {"ADDRESS", "LABEL"}

    os.unlink(gt_file.name)
    os.unlink(sub_file.name)

def test_load_data_skip_column_check_with_split():
    # Create ground truth with split and extra column
    ground_truth = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.0, 2.0],
        "SPLIT": ["public", "private"],
        "EXTRA": ["X", "Y"]
    })
    # Create submission with different columns
    submission = pd.DataFrame({
        "ADDRESS": ["ADDR1", "ADDR2"],
        "LABEL": [1.1, 2.1]
    })

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as gt_file:
        ground_truth.to_parquet(gt_file.name)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        submission.to_csv(sub_file.name, index=False)

    # Should succeed with skip_column_check and custom_split
    gt_df, sub_df = load_data(gt_file.name, sub_file.name, custom_split="public", skip_column_check=True)
    assert "SPLIT" in gt_df.columns
    assert "EXTRA" in gt_df.columns
    public_rows = gt_df[gt_df["SPLIT"].str.lower() == "public"]
    assert len(public_rows) == 1  # Only public split
    assert public_rows["SPLIT"].iloc[0].lower() == "public"

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

# Test process_pairwise_data
@pytest.fixture
def sample_pairwise_data():
    ground_truth = pd.DataFrame([
        ["src1", "src2", "quality", 0.5],
        ["src2", "src3", "quality", 0.8],
        ["src3", "src4", "originality", 1.2]
    ], columns=["SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"])
    submission = pd.DataFrame([
        ["src1", "quality", 1.0],
        ["src2", "quality", 2.0],
        ["src3", "quality", 3.0],
        ["src3", "originality", 4.0]
    ], columns=["SOURCE", "TARGET", "WEIGHT"])
    return ground_truth, submission

@pytest.fixture
def sample_pairwise_split_data():
    ground_truth = pd.DataFrame([
        ["src1", "src2", "quality", 0.5, "public"],
        ["src2", "src3", "quality", 0.8, "public"],
        ["src3", "src4", "originality", 1.2, "private"],
        ["src4", "src5", "quality", 1.5, "private"]
    ], columns=["SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A", "SPLIT"])
    submission = pd.DataFrame([
        ["src1", "quality", 1.0],
        ["src2", "quality", 2.0],
        ["src3", "quality", 3.0],
        ["src3", "originality", 4.0],
        ["src4", "quality", 5.0],
        ["src5", "quality", 6.0]
    ], columns=["SOURCE", "TARGET", "WEIGHT"])
    return ground_truth, submission

def test_process_pairwise_data_valid(sample_pairwise_data):
    ground_truth, submission = sample_pairwise_data
    gt_processed, sub_processed = process_pairwise_data(ground_truth, submission)
    
    assert isinstance(gt_processed, pd.DataFrame)
    assert isinstance(sub_processed, pd.DataFrame)
    assert set(gt_processed.columns) == {"SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"}
    assert set(sub_processed.columns) == {"SOURCE", "TARGET", "WEIGHT"}
    assert len(gt_processed) == 3
    assert len(sub_processed) == 4
    
    # Check case conversion
    assert gt_processed["SOURCE_A"].str.isupper().all()
    assert gt_processed["SOURCE_B"].str.isupper().all()
    assert gt_processed["TARGET"].str.isupper().all()
    assert sub_processed["SOURCE"].str.isupper().all()
    assert sub_processed["TARGET"].str.isupper().all()

def test_process_pairwise_data_with_split(sample_pairwise_split_data):
    ground_truth, submission = sample_pairwise_split_data
    gt_processed, sub_processed = process_pairwise_data(ground_truth, submission, custom_split="public")
    
    assert len(gt_processed) == 2  # Only public split rows
    assert all(row["TARGET"] == "QUALITY" for _, row in gt_processed.iterrows())
    
    # Test private split
    gt_processed, _ = process_pairwise_data(ground_truth, submission, custom_split="private")
    assert len(gt_processed) == 2  # Only private split rows

def test_process_pairwise_data_invalid_split(sample_pairwise_split_data):
    ground_truth, submission = sample_pairwise_split_data
    with pytest.raises(ValueError, match="custom_split must be either 'public' or 'private'"):
        process_pairwise_data(ground_truth, submission, custom_split="invalid")

def test_process_pairwise_data_duplicate_predictions():
    ground_truth = pd.DataFrame([
        ["src1", "src2", "quality", 0.5]
    ], columns=["SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"])
    submission = pd.DataFrame([
        ["src1", "quality", 1.0],
        ["src1", "quality", 2.0]  # Duplicate
    ], columns=["SOURCE", "TARGET", "WEIGHT"])
    
    with pytest.raises(ValueError, match="Submission contains duplicate rows for SOURCE and TARGET combinations"):
        process_pairwise_data(ground_truth, submission)

def test_process_pairwise_data_case_insensitive(sample_pairwise_data):
    ground_truth, submission = sample_pairwise_data
    # Modify case in both dataframes
    ground_truth.loc[0, "SOURCE_A"] = "SRC1"
    ground_truth.loc[0, "TARGET"] = "QUALITY"
    submission.loc[0, "SOURCE"] = "src1"
    submission.loc[0, "TARGET"] = "quality"
    
    gt_processed, sub_processed = process_pairwise_data(ground_truth, submission)
    
    # Check case normalization
    assert gt_processed.loc[0, "SOURCE_A"] == "SRC1"
    assert gt_processed.loc[0, "TARGET"] == "QUALITY"
    assert sub_processed.loc[0, "SOURCE"] == "SRC1"
    assert sub_processed.loc[0, "TARGET"] == "QUALITY"

# Test load_process_data_deepfunding
@pytest.fixture
def sample_deepfunding_data():
    # Create ground truth data
    ground_truth = pd.DataFrame({
        "SOURCE_A": ["a", "b", "c"],
        "SOURCE_B": ["b", "c", "a"],
        "TARGET": ["t", "t", "t"],
        "B_OVER_A": [2.0, 1.5, 0.5]
    })
    
    # Create submission data files
    submission1 = pd.DataFrame({
        "SOURCE": ["a", "b", "c"],
        "TARGET": ["t", "t", "t"],
        "WEIGHT": [1.0, 2.0, 1.5]
    })
    submission2 = pd.DataFrame({
        "SOURCE": ["a", "b", "c"],
        "TARGET": ["t", "t", "t"],
        "WEIGHT": [1.2, 2.0, 3.0]
    })
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as gt_file, \
         tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as sub_paths_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub1_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub2_file:
        
        # Write ground truth
        ground_truth.to_csv(gt_file.name, index=False)
        
        # Write submissions
        submission1.to_csv(sub1_file.name, index=False)
        submission2.to_csv(sub2_file.name, index=False)
        
        # Write submission paths
        with open(sub_paths_file.name, 'w') as f:
            f.write(f"{sub1_file.name}\n{sub2_file.name}\n")
            
        yield gt_file.name, sub_paths_file.name
        
        # Cleanup
        os.unlink(gt_file.name)
        os.unlink(sub_paths_file.name)
        os.unlink(sub1_file.name)
        os.unlink(sub2_file.name)

@pytest.fixture
def sample_deepfunding_split_data():
    # Create ground truth data with split
    ground_truth = pd.DataFrame({
        "SOURCE_A": ["a", "b", "c", "d"],
        "SOURCE_B": ["b", "c", "a", "a"],
        "TARGET": ["t", "t", "t", "t"],
        "B_OVER_A": [2.0, 1.5, 0.5, 1.0],
        "SPLIT": ["public", "public", "private", "private"]
    })
    
    # Create submission data files
    submission1 = pd.DataFrame({
        "SOURCE": ["a", "b", "c", "d"],
        "TARGET": ["t", "t", "t", "t"],
        "WEIGHT": [1.0, 2.0, 1.5, 2.0]
    })
    submission2 = pd.DataFrame({
        "SOURCE": ["a", "b", "c", "d"],
        "TARGET": ["t", "t", "t", "t"],
        "WEIGHT": [1.2, 2.0, 3.0, 1.5]
    })
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as gt_file, \
         tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as sub_paths_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub1_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub2_file:
        
        # Write ground truth
        ground_truth.to_csv(gt_file.name, index=False)
        
        # Write submissions
        submission1.to_csv(sub1_file.name, index=False)
        submission2.to_csv(sub2_file.name, index=False)
        
        # Write submission paths
        with open(sub_paths_file.name, 'w') as f:
            f.write(f"{sub1_file.name}\n{sub2_file.name}\n")
            
        yield gt_file.name, sub_paths_file.name
        
        # Cleanup
        os.unlink(gt_file.name)
        os.unlink(sub_paths_file.name)
        os.unlink(sub1_file.name)
        os.unlink(sub2_file.name)

def test_load_process_data_deepfunding_valid(sample_deepfunding_data):
    gt_path, sub_paths = sample_deepfunding_data
    ground_truth, submissions = load_process_data_deepfunding(gt_path, sub_paths)
    
    # Check ground truth data
    assert isinstance(ground_truth, pd.DataFrame)
    assert set(ground_truth.columns) == {"SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"}
    assert ground_truth["SOURCE_A"].str.isupper().all()
    assert ground_truth["SOURCE_B"].str.isupper().all()
    assert ground_truth["TARGET"].str.isupper().all()
    assert ground_truth["B_OVER_A"].dtype == float
    
    # Check submissions
    assert isinstance(submissions, list)
    assert len(submissions) == 2
    for submission in submissions:
        assert isinstance(submission, pd.DataFrame)
        assert set(submission.columns) == {"SOURCE", "TARGET", "WEIGHT"}
        assert submission["SOURCE"].str.isupper().all()
        assert submission["TARGET"].str.isupper().all()
        assert submission["WEIGHT"].dtype == float
        assert not submission["WEIGHT"].isnull().any()

def test_load_process_data_deepfunding_with_split(sample_deepfunding_split_data):
    gt_path, sub_paths = sample_deepfunding_split_data
    ground_truth, submissions = load_process_data_deepfunding(gt_path, sub_paths, custom_split="public")
    
    # Check that only public data is included
    assert len(ground_truth) == 2  # Only public rows
    assert ground_truth["B_OVER_A"].tolist() == [2.0, 1.5]  # Public values
    
    # Check that submissions only include sources from public split
    for submission in submissions:
        assert len(submission) == 3  # Only sources from public comparisons (a, b, c)
        assert set(submission["SOURCE"].tolist()) == {"A", "B", "C"}

def test_load_process_data_deepfunding_invalid_split(sample_deepfunding_split_data):
    gt_path, sub_paths = sample_deepfunding_split_data
    with pytest.raises(ValueError, match="custom_split must be either 'public' or 'private'"):
        load_process_data_deepfunding(gt_path, sub_paths, custom_split="invalid")

def test_load_process_data_deepfunding_missing_weights():
    # Create ground truth data
    ground_truth = pd.DataFrame({
        "SOURCE_A": ["a", "b"],
        "SOURCE_B": ["b", "c"],
        "TARGET": ["t", "t"],
        "B_OVER_A": [2.0, 1.5]
    })
    
    # Create submission with missing source
    submission = pd.DataFrame({
        "SOURCE": ["a", "b"],  # Missing 'c'
        "TARGET": ["t", "t"],
        "WEIGHT": [1.0, 2.0]
    })
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as gt_file, \
         tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as sub_paths_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as sub_file:
        
        ground_truth.to_csv(gt_file.name, index=False)
        submission.to_csv(sub_file.name, index=False)
        
        with open(sub_paths_file.name, 'w') as f:
            f.write(f"{sub_file.name}\n")
        
        with pytest.raises(ValueError, match="Missing weights in submission data"):
            load_process_data_deepfunding(gt_file.name, sub_paths_file.name)
        
        os.unlink(gt_file.name)
        os.unlink(sub_paths_file.name)
        os.unlink(sub_file.name)

def test_load_process_data_deepfunding_case_insensitive(sample_deepfunding_data):
    gt_path, sub_paths = sample_deepfunding_data
    
    # Load the files
    ground_truth = pd.read_csv(gt_path)
    ground_truth.loc[0, "SOURCE_A"] = ground_truth.loc[0, "SOURCE_A"].lower()
    ground_truth.loc[1, "SOURCE_B"] = ground_truth.loc[1, "SOURCE_B"].lower()
    ground_truth.loc[0, "TARGET"] = ground_truth.loc[0, "TARGET"].lower()
    ground_truth.to_csv(gt_path, index=False)
    
    # Load and process
    ground_truth, submissions = load_process_data_deepfunding(gt_path, sub_paths)
    
    # Check case conversion
    assert ground_truth["SOURCE_A"].str.isupper().all()
    assert ground_truth["SOURCE_B"].str.isupper().all()
    assert ground_truth["TARGET"].str.isupper().all()
    
    for submission in submissions:
        assert submission["SOURCE"].str.isupper().all()
        assert submission["TARGET"].str.isupper().all()
