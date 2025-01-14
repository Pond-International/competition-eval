# Competition Evaluation

A Python package for evaluating competition submissions.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables by copying `.env.example` to `.env` and filling in your credentials if you want to use AWS S3:
```bash
cp .env.example .env
# Edit .env with your database and AWS credentials
```

## Usage

The `evaluate.py` script compares predictions against ground truth using various metrics:

```bash
python evaluate.py <ground_truth_file> <submission_file> <metric> [--topk K] [--data-portion PORTION] [--after-split] [--custom-split SPLIT_TYPE]
```

Arguments:
- `ground_truth_file`: Path to ground truth parquet or csv file. Can be local path or S3 URL.
- `submission_file`: Path to submission CSV file. Can be local path or S3 URL.
- `metric`: Metric to compute (see Available Metrics below)
- `--topk`: Number of top recommendations to consider for precision_at_k metric (default: 5)
- `--data-portion`: Portion of ground truth data to use (float between 0 and 1, default: 1.0)
- `--after-split`: If set, use remaining (1-portion) of data after the split point
- `--custom-split`: Customized split ('public' or 'private') of ground truth data to use for evaluation. Cannot be used with `--data-portion`. If None, no split is applied. Default: None

Available metrics:
- `accuracy`: Classification accuracy
- `rmse`: Root mean squared error
- `wmse`: Weighted mean squared error (weighted by absolute ground truth value)
- `precision_at_k`: Precision at k for ranking tasks
- `dcg`: Discounted Cumulative Gain for ranking tasks (evaluates both the rank position and relevance of recommendations)
- `auc`: Area Under the Curve for binary classification tasks
- `mse`: Mean squared error

Example:
```bash
# Evaluate using all data
python evaluate.py data/123_ground_truth.parquet data/123_1_dev1.csv rmse

# Evaluate using first 75% of data
python evaluate.py data/123_ground_truth.parquet data/123_1_dev1.csv rmse --data-portion 0.75

# Evaluate using remaining 25% of data after 75% split point
python evaluate.py data/123_ground_truth.parquet data/123_1_dev1.csv rmse --data-portion 0.75 --after-split

# Evaluate using public split
python evaluate.py data/123_ground_truth.parquet data/123_1_dev1.csv rmse --custom-split public
```

Output:

Evaluation results are printed to the console in the following format:

- If submission is valid: 
  ```json
  Final result: {"submission_url": "s3://submissions/123_1_dev1.csv", "status": 200, "error_reason": "", "final_result": 0.996}
  ```

- If submission is invalid:
  ```json
  Final result: {"submission_url": "s3://submissions/123_1_dev1.csv", "status": 400, "error_reason": "Input arrays cannot contain missing values (NaN)", "final_result": ""}
  ```

## Development

### General Procedure
1. Create a new branch for your changes
2. Make changes and add new tests into the `tests/` directory. `Pytest` is used for testing.
3. Run `pytest tests/` to make sure all tests are passing
4. Commit changes
5. Push changes to GitHub
6. Create a pull request
7. Request a review from a code reviewer
8. Merge the pull request once review is complete

### Running Tests
```bash
pytest tests/
```

### Adding New Metrics
1. Add the metric function to `metrics.py`
2. Add corresponding unit tests in `tests/test_metrics.py`
3. Register the metric in the `METRICS` dictionary in `metrics.py`



