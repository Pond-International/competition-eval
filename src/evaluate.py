#!/usr/bin/env python3
"""
Main evaluation script for comparing predictions against ground truth.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Tuple, Union, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from metrics import METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(
    ground_truth_path: str, submission_path: str, custom_split: str = None, skip_column_check: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ground truth and submission data.

    Args:
        ground_truth_path: Path to ground truth parquet file
        submission_path: Path to submission CSV file
        custom_split: Optional split type ('public' or 'private') to filter data

    Returns:
        Tuple of (ground_truth_df, submission_df)
    """
    logger.debug(f"Loading ground truth data from {ground_truth_path}")
    if ground_truth_path.endswith("parquet"):
        ground_truth_df = pd.read_parquet(ground_truth_path)
    elif ground_truth_path.endswith("csv"):
        ground_truth_df = pd.read_csv(ground_truth_path)
    else:
        raise ValueError(f"Unsupported file type for ground truth: {ground_truth_path}")
    logger.debug(f"Ground truth shape: {ground_truth_df.shape}")

    logger.debug(f"Loading submission data from {submission_path}")
    submission_df = pd.read_csv(submission_path)
    logger.debug(f"Submission shape: {submission_df.shape}")

    # Get columns without the split column if it exists
    gt_cols = [col for col in ground_truth_df.columns if col.lower() != 'split']
    
    if not skip_column_check and not np.array_equal(gt_cols, submission_df.columns):
        raise ValueError("Unexpected column names in submission")

    if custom_split is not None and 'split' not in [col.lower() for col in ground_truth_df.columns]:
        raise ValueError("Ground truth data does not contain a 'split' column")

    return ground_truth_df, submission_df

def load_process_data_deepfunding(ground_truth_path: str, submission_paths: str, custom_split: str = None) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    logger.debug(f"Loading ground truth data from {ground_truth_path}")
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    # Check required columns
    if custom_split is not None and 'split' not in [col.lower() for col in ground_truth_df.columns]:
        raise ValueError("Ground truth data does not contain a 'split' column")
    cols = ["SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"]
    if custom_split is not None:
        cols.append("SPLIT")
    ground_truth_df.columns = cols
    ground_truth_df["SOURCE_A"] = ground_truth_df["SOURCE_A"].astype(str)
    ground_truth_df["SOURCE_A"] = ground_truth_df["SOURCE_A"].str.upper()
    ground_truth_df["SOURCE_B"] = ground_truth_df["SOURCE_B"].astype(str)
    ground_truth_df["SOURCE_B"] = ground_truth_df["SOURCE_B"].str.upper()
    ground_truth_df["TARGET"] = ground_truth_df["TARGET"].astype(str)
    ground_truth_df["TARGET"] = ground_truth_df["TARGET"].str.upper()
    ground_truth_df["B_OVER_A"] = ground_truth_df["B_OVER_A"].astype(float)

    # Filter out originality scores
    ground_truth_df = ground_truth_df[ground_truth_df["TARGET"]!="ORIGINALITY"]

    # Filter by split if specified
    if custom_split is not None:
        split_value = custom_split.lower()
        if split_value not in ['public', 'private']:
            raise ValueError("custom_split must be either 'public' or 'private'")
        ground_truth_df = ground_truth_df[ground_truth_df['SPLIT'].str.lower() == split_value]
        logger.debug(f"Filtered to {split_value} split, new shape: {ground_truth_df.shape}")

    logger.debug(f"Ground truth shape: {ground_truth_df.shape}")

    source_target = pd.concat(
        [
            ground_truth_df[["SOURCE_A", "TARGET"]].rename(columns={"SOURCE_A": "SOURCE"}),
            ground_truth_df[["SOURCE_B", "TARGET"]].rename(columns={"SOURCE_B": "SOURCE"})
        ],
        axis=0
    )
    source_target = source_target.drop_duplicates()

    # Load all submissions from their paths
    logger.debug(f"Loading submissions from {submission_paths}")
    all_submissions = pd.read_csv(submission_paths)
    submissions = []
    for submission_path in all_submissions["path"]:
        submission_df = pd.read_csv(submission_path.strip())
        submission_df.columns = ["SOURCE", "TARGET", "WEIGHT"]
        submission_df["SOURCE"] = submission_df["SOURCE"].astype(str)
        submission_df["SOURCE"] = submission_df["SOURCE"].str.upper()
        submission_df["TARGET"] = submission_df["TARGET"].astype(str)
        submission_df["TARGET"] = submission_df["TARGET"].str.upper()
        submission_df["WEIGHT"] = submission_df["WEIGHT"].astype(float)
        submission_df = source_target.merge(submission_df, on=["SOURCE", "TARGET"], how="left")
        # Check for missing weights
        if submission_df["WEIGHT"].isnull().any():
            raise ValueError("Missing weights in submission data")
        # Log-transform weights
        zero_weights = submission_df["WEIGHT"] == 0
        submission_df.loc[zero_weights, "WEIGHT"] = np.log(1e-18)
        submission_df.loc[~zero_weights, "WEIGHT"] = np.log(submission_df.loc[~zero_weights, "WEIGHT"])
        submission_df.sort_values(by=["TARGET","SOURCE"], inplace=True)
        submissions.append(submission_df)
    logger.debug(f"Number of submissions: {len(submissions)}")

    return ground_truth_df, submissions


def process_supervised_data(
    ground_truth_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    data_portion: float = 1.0,
    after_split: bool = False,
    custom_split: str = None,
) -> pd.DataFrame:
    """Validate that ground truth and submission dataframes are compatible.

    Args:
        ground_truth_df: Ground truth dataframe
        submission_df: Submission dataframe
        data_portion: Portion of data to use (between 0 and 1)
        after_split: If True, use remaining portion after split point
        custom_split: Optional split type ('public' or 'private') to filter ground truth data

    Returns:
        pd.DataFrame: Combined and validated dataframe

    Raises:
        ValueError: If validation fails or if data_portion is invalid
    """
    if after_split and data_portion >= 1:
        raise ValueError("data-portion must be less than 1 when after_split is enabled")
    if not 0 < data_portion <= 1:
        raise ValueError("data-portion must be between 0 and 1")

    logger.debug("Validating supervised input data")

    # Check required columns
    logger.debug("Selecting required columns")
    cols = ["ADDRESS", "LABEL"]
    if custom_split is not None:
        cols.append("SPLIT")
    ground_truth_df.columns = cols
    ground_truth_df["ADDRESS"] = ground_truth_df["ADDRESS"].astype(str)
    ground_truth_df["ADDRESS"] = ground_truth_df["ADDRESS"].str.upper()
    ground_truth_df["LABEL"] = pd.to_numeric(ground_truth_df["LABEL"])

    submission_df.columns = ["ADDRESS", "PRED"]
    submission_df["ADDRESS"] = submission_df["ADDRESS"].astype(str)
    submission_df["ADDRESS"] = submission_df["ADDRESS"].str.upper()
    submission_df["PRED"] = pd.to_numeric(submission_df["PRED"])

    # Check for duplicate addresses
    logger.debug("Checking for duplicate addresses in submission data")
    if submission_df["ADDRESS"].nunique() != len(submission_df):
        raise ValueError("Duplicate addresses found in submission data")

    logger.debug("Merging ground truth and submission data")
    df = ground_truth_df.merge(submission_df, how="left", on="ADDRESS")
    logger.debug(f"Merged dataframe shape: {df.shape}")

    # Filter by split if specified
    if custom_split is not None:
        split_value = custom_split.lower()
        if split_value not in ['public', 'private']:
            raise ValueError("custom_split must be either 'public' or 'private'")
        df = df[df['SPLIT'].str.lower() == split_value]
        logger.debug(f"Filtered to {split_value} split, new shape: {df.shape}")

    if data_portion < 1.0:
        n_samples = int(len(df) * data_portion)
        if after_split:
            df = df.iloc[n_samples:]
            logger.debug(
                f"Selected remaining {1-data_portion:.2%} of ground truth after {data_portion:.2%} split, new shape: {df.shape}"
            )
        else:
            df = df.iloc[:n_samples]
            logger.debug(
                f"Selected first {data_portion:.2%} of ground truth, new shape: {df.shape}"
            )

    logger.debug(f"Final dataframe:\n{df.head(5)}")

    return df


def process_recommend_data(
    ground_truth_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    data_portion: float = 1.0,
    after_split: bool = False,
    custom_split: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that ground truth and submission dataframes are compatible.

    Args:
        ground_truth_df: Ground truth dataframe
        submission_df: Submission dataframe
        data_portion: Portion of data to use (between 0 and 1)
        after_split: If True, use remaining portion after split point
        custom_split: Optional split type ('public' or 'private') to filter data

    Returns:
        pd.DataFrame: Combined and validated dataframe

    Raises:
        ValueError: If validation fails or if data_portion is invalid
    """
    if after_split and data_portion >= 1:
        raise ValueError("data-portion must be less than 1 when after_split is enabled")
    if not 0 < data_portion <= 1:
        raise ValueError("data-portion must be between 0 and 1")

    logger.debug("Validating recommendation input data")

    # Check required columns
    logger.debug("Selecting required columns")
    cols = ["ADDRESS", "REC"]
    if custom_split is not None:
        cols.append("SPLIT")
    ground_truth_df.columns = cols
    ground_truth_df["ADDRESS"] = ground_truth_df["ADDRESS"].astype(str)
    ground_truth_df["ADDRESS"] = ground_truth_df["ADDRESS"].str.upper()
    ground_truth_df["REC"] = ground_truth_df["REC"].astype(str)
    ground_truth_df["REC"] = ground_truth_df["REC"].str.upper()

    submission_df.columns = ["ADDRESS", "REC"]
    submission_df["ADDRESS"] = submission_df["ADDRESS"].astype(str)
    submission_df["ADDRESS"] = submission_df["ADDRESS"].str.upper()
    submission_df["REC"] = submission_df["REC"].astype(str)
    submission_df["REC"] = submission_df["REC"].str.upper()

    # Filter by split if specified
    if custom_split is not None:
        split_value = custom_split.lower()
        if split_value not in ['public', 'private']:
            raise ValueError("custom_split must be either 'public' or 'private'")
        ground_truth_df = ground_truth_df[ground_truth_df['SPLIT'].str.lower() == split_value]
        logger.debug(f"Filtered to {split_value} split, new shape: {ground_truth_df.shape}")

    if data_portion < 1.0:
        addresses = ground_truth_df["ADDRESS"].drop_duplicates()
        n_samples = int(len(addresses) * data_portion)
        if after_split:
            addresses = addresses.iloc[n_samples:]
            logger.debug(
                f"Selected remaining {1-data_portion:.2%} of unique addresses, length: {len(addresses)}"
            )
        else:
            addresses = addresses.iloc[:n_samples]
            logger.debug(
                f"Selected first {data_portion:.2%} of unique addresses, length: {len(addresses)}"
            )

        ground_truth_df = ground_truth_df.merge(addresses, on="ADDRESS")
        submission_df = submission_df.merge(addresses, on="ADDRESS")
        logger.debug(
            f"Ground truth new shape: {ground_truth_df.shape}, submission new shape: {submission_df.shape}"
        )

    return ground_truth_df, submission_df

def process_pairwise_data(
    ground_truth_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    custom_split: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that ground truth and submission dataframes are compatible.

    Args:
        ground_truth_df: Ground truth dataframe
        submission_df: Submission dataframe
        custom_split: Optional split type ('public' or 'private') to filter data

    Returns:
        pd.DataFrame: Combined and validated dataframe

    Raises:
        ValueError: If validation fails
    """

    logger.debug("Validating pairwise ranking input data")

    # Check required columns
    logger.debug("Selecting required columns")
    cols = ["SOURCE_A", "SOURCE_B", "TARGET", "B_OVER_A"]
    if custom_split is not None:
        cols.append("SPLIT")
    ground_truth_df.columns = cols
    ground_truth_df["SOURCE_A"] = ground_truth_df["SOURCE_A"].astype(str)
    ground_truth_df["SOURCE_A"] = ground_truth_df["SOURCE_A"].str.upper()
    ground_truth_df["SOURCE_B"] = ground_truth_df["SOURCE_B"].astype(str)
    ground_truth_df["SOURCE_B"] = ground_truth_df["SOURCE_B"].str.upper()
    ground_truth_df["TARGET"] = ground_truth_df["TARGET"].astype(str)
    ground_truth_df["TARGET"] = ground_truth_df["TARGET"].str.upper()
    ground_truth_df["B_OVER_A"] = ground_truth_df["B_OVER_A"].astype(float)

    submission_df.columns = ["SOURCE", "TARGET", "WEIGHT"]
    submission_df["SOURCE"] = submission_df["SOURCE"].astype(str)
    submission_df["SOURCE"] = submission_df["SOURCE"].str.upper()
    submission_df["TARGET"] = submission_df["TARGET"].astype(str)
    submission_df["TARGET"] = submission_df["TARGET"].str.upper()
    submission_df["WEIGHT"] = submission_df["WEIGHT"].astype(float)
    # Check if all WEIGHT values are non-negative
    if (submission_df["WEIGHT"] < 0).any():
        raise ValueError("All weight values in the submission must be non-negative.")

    zero_weights = submission_df["WEIGHT"] == 0
    submission_df.loc[zero_weights, "WEIGHT"] = np.log(1e-18)
    submission_df.loc[~zero_weights, "WEIGHT"] = np.log(submission_df.loc[~zero_weights, "WEIGHT"])

    # Check for duplicate rows based on SOURCE and TARGET columns
    duplicate_rows = submission_df[submission_df.duplicated(subset=['SOURCE', 'TARGET'], keep=False)]
    if not duplicate_rows.empty:
        raise ValueError("Submission contains duplicate rows for SOURCE and TARGET combinations.")

    # Filter by split if specified
    if custom_split is not None:
        split_value = custom_split.lower()
        if split_value not in ['public', 'private']:
            raise ValueError("custom_split must be either 'public' or 'private'")
        ground_truth_df = ground_truth_df[ground_truth_df['SPLIT'].str.lower() == split_value]
        logger.debug(f"Filtered to {split_value} split, new shape: {ground_truth_df.shape}")

    return ground_truth_df, submission_df

def compute_metric(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    metric_name: str,
    metric_params: Dict[str, Any] = None,
) -> float:
    """Compute specified metric between ground truth and submission.

    Args:
        y_true: Ground truth
        y_pred: predictions
        metric_name: Name of metric to compute
        metric_params: Optional parameters for metric computation

    Returns:
        float: Computed metric value

    Raises:
        ValueError: If metric_name is not recognized
    """
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available metrics: {list(METRICS.keys())}"
        )

    logger.debug(f"Computing {metric_name}")
    if metric_params:
        logger.debug(f"Using parameters: {metric_params}")

    metric_fn = METRICS[metric_name]
    if metric_params is None:
        metric_params = {}

    score = metric_fn(y_true, y_pred, **metric_params)
    logger.debug(f"{metric_name} score: {score:.6f}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth."
    )
    parser.add_argument(
        "ground_truth_path", help="Path to ground truth parquet or CSV file"
    )
    parser.add_argument("submission_path", help="Path to submission CSV file")
    parser.add_argument(
        "metric_name",
        help=f"Metric to compute. Available metrics: {list(METRICS.keys())}",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top recommendations to consider for precision_at_k metric",
    )
    parser.add_argument(
        "--data-portion",
        type=float,
        default=1.0,
        help="Portion of ground truth data to use for evaluation (between 0 and 1)",
    )
    parser.add_argument(
        "--after-split",
        action="store_true",
        help="If true, use remaining (1-data_portion) of data after the split point instead of the first data_portion",
    )
    parser.add_argument(
        "--custom-split",
        type=str,
        choices=['public', 'private'],
        help="Filter data by split type (public or private)",
    )
    parser.add_argument(
        "--skip-column-check",
        action="store_true",
        help="Skip checking whether column names match between ground truth and submission",
    )
    args = parser.parse_args()

    if args.data_portion != 1.0 and args.custom_split is not None:
        parser.error("--data-portion and --custom-split cannot be used together")

    # Load environment variables
    if not load_dotenv():
        logger.warning("No .env file found.")

    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}. Will not be able to read from AWS S3.")

    if args.after_split and args.data_portion >= 1:
        raise ValueError(
            "data-portion must be less than 1 when --after-split is enabled"
        )
    if not 0 < args.data_portion <= 1:
        raise ValueError("data-portion must be between 0 and 1")

    try:
        logger.info("Starting evaluation")
        logger.info(f"Metric: {args.metric_name}")
        if args.after_split:
            logger.info(
                f"Using remaining {1-args.data_portion:.2%} of ground truth data after {args.data_portion:.2%} split"
            )
        else:
            logger.info(f"Using first {args.data_portion:.2%} of ground truth data")

        metric_params = {}
        # Load data
        if args.metric_name == "deepfunding":
            y_true, y_pred = load_process_data_deepfunding(
                args.ground_truth_path, args.submission_path, args.custom_split
            )
        else:
            ground_truth_df, submission_df = load_data(
                args.ground_truth_path, args.submission_path, args.custom_split,
                args.skip_column_check
            )
            if args.metric_name == "dcg":
                if submission_df["RANK"].max() > len(ground_truth_df):
                    raise ValueError("Some ranks are too large.")

        # Validate data
        if args.metric_name == "precision_at_k":
            y_true, y_pred = process_recommend_data(
                ground_truth_df, submission_df, args.data_portion, args.after_split, args.custom_split
            )
            metric_params["k"] = args.topk
        elif args.metric_name == "pairwise_cost":
            y_true, y_pred = process_pairwise_data(
                ground_truth_df, submission_df, args.custom_split
            )
        else:
            combined_df = process_supervised_data(
                ground_truth_df, submission_df, args.data_portion, args.after_split, args.custom_split
            )
            y_true = combined_df["LABEL"].values
            y_pred = combined_df["PRED"].values

        # Compute metric
        score = compute_metric(y_true, y_pred, args.metric_name, metric_params)

        logger.info("Evaluation completed successfully")

        ret = {
            "submission_url": args.submission_path,
            "status": 200,
            "error_reason": "",
            "final_result": score,
        }
        print("Final result: " + json.dumps(ret))
        exit(0)
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        ret = {
            "submission_url": args.submission_path,
            "status": 400,
            "error_reason": str(e),
            "final_result": "",
        }
        print("Final result: " + json.dumps(ret))
        exit(1)


if __name__ == "__main__":
    main()
