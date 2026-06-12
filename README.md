# Home Health Visit Delay Prediction

This folder contains a complete Eris-style ML challenge draft:

- `raw/generate_raw.py` creates a deterministic synthetic raw dataset.
- `dataset_description.md` documents the raw data.
- `prepare.py` builds `public/train.csv`, `public/test.csv`, `public/sample_submission.csv`, and `private/answers.csv`.
- `problem.md` is the challenge statement shown to solvers.
- `grade.py` validates submissions and computes the private score.
- `config.yaml` defines task metadata and score direction.
- `rubrics.yaml` contains task-specific rubric criteria.
- `reference_solution.py` is a solvability check that runs end to end on the prepared public data.
- `solution.ipynb` is the notebook version to upload as the reference solution.

The task is synthetic, so there are no redistribution or commercial-use license issues. It is designed to test practical tabular ML judgment: missing values, skewed regression targets, time features, service-region effects, weather and traffic signals, categorical interactions, and leakage avoidance.

## Submission Mapping

1. Create the dataset first.
   - Upload `raw/data.csv` as the raw data.
   - Paste `dataset_description.md` into the dataset description.
   - Upload `prepare.py` as the prepare script.
   - Run the platform checks and confirm it creates `train.csv`, `test.csv`, `sample_submission.csv`, and `answers.csv`.

2. Create the challenge from that dataset.
   - Paste `problem.md` as the problem description.
   - Upload `grade.py` as the grading script.
   - Use `config.yaml` for name, domain, difficulty, and score direction.
   - Add the criteria from `rubrics.yaml`.

3. Submit the reference solution.
   - Upload `solution.ipynb`.
   - It reads from `/dataset/public`, writes `/working/submission.csv`, and uses only standard pandas/numpy libraries.
