# House Prices: Advanced Regression

A configuration-driven machine learning project for predicting residential property prices using the Kaggle **House Prices: Advanced Regression Techniques** competition.

The project explores an end-to-end regression workflow, including exploratory data analysis, preprocessing, model benchmarking, cross-validation, hyperparameter optimization, model evaluation, and Kaggle submission generation.

## Project Objectives

- Build a modular and reusable machine learning workflow.
- Keep experiment settings and pipeline parameters in a YAML configuration file.
- Compare different regression algorithms using consistent evaluation procedures.
- Use cross-validation and Optuna for hyperparameter optimization.
- Evaluate model performance using Root Mean Squared Logarithmic Error (RMSLE).
- Generate predictions in the format required by the Kaggle competition.

## Dataset

The project uses the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset.

| Property | Description |
|---|---|
| Training observations | 1,460 |
| Test observations | 1,459 |
| Training columns | 81 |
| Test columns | 80 |
| Target variable | `SalePrice` |
| Identifier | `Id` |
| Task | Supervised regression |

The dataset contains numerical and categorical property characteristics. The target variable, `SalePrice`, represents the property's sale price.

**Dataset access:** The project uses the Kaggle API. You must have access to the competition data and configure your Kaggle credentials before downloading it.

## Project Structure

```text
advanced_regression_housing_price/
├── config/
│   └── config.yaml
├── notebook/
│   └── eda.ipynb
├── reports/
│   ├── best_hyperparameters.json
│   ├── cross_validation_results.csv
│   ├── final_model_metadata.json
│   ├── optuna_studies.db
│   └── optuna_trials.csv
├── src/
│   ├── config_loader.py
│   ├── cross_validation.py
│   ├── diagnostic.py
│   ├── evaluate_tuned_model.py
│   ├── evaluation.py
│   ├── finalize_model.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── models.py
│   ├── playground.py
│   ├── preprocessor.py
│   ├── split.py
│   ├── submission.py
│   ├── train.py
│   └── tuning.py
├── requirements.txt
└── .gitignore
```

Generated datasets, model artifacts, logs, and other outputs may be stored locally and are excluded from version control according to `.gitignore`.

## Workflow and Architecture

The project separates major responsibilities into modules to make the workflow easier to maintain and extend.

### 1. Configuration Management

`config/config.yaml` stores project settings, while `src/config_loader.py` provides configuration-loading functionality.

This approach reduces the need to hardcode experiment settings throughout the codebase.

### 2. Data Loading and Splitting

- `src/load_data.py` handles dataset loading through the Kaggle API.
- `src/split.py` handles the data-splitting logic.

The competition's test dataset does not include the target variable, so it is used for generating submission predictions rather than evaluating model performance against known labels.

### 3. Exploratory Data Analysis

`notebook/eda.ipynb` contains the exploratory analysis of the dataset, including investigation of feature types, missing values, relationships between variables, and potential preprocessing requirements.

### 4. Preprocessing

`src/preprocessor.py` contains the preprocessing components and transformer construction.

The project uses scikit-learn preprocessing tools, including imputers, encoders, column transformations, and pipelines. Keeping preprocessing within the machine learning workflow helps maintain consistent transformations between training and prediction.

### 5. Model Benchmarking

`src/models.py` defines the candidate models, and `src/train.py` handles model-training functionality.

The candidate algorithms include:

- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Additional scikit-learn regression estimators are available in the model definitions.

### 6. Cross-Validation

`src/cross_validation.py` implements cross-validation functionality, while `src/evaluation.py` handles model evaluation.

Cross-validation provides a more robust comparison than relying on a single validation split. The primary competition metric is RMSLE, for which lower values indicate better performance.

### 7. Hyperparameter Optimization

`src/tuning.py` contains the Optuna-based tuning workflow.

The tuning results and experiment artifacts are stored in the `reports/` directory, including the best hyperparameters, trial results, and Optuna study database.

### 8. Final Model and Diagnostics

- `src/evaluate_tuned_model.py` evaluates tuned-model performance.
- `src/finalize_model.py` handles final-model preparation.
- `src/diagnostic.py` provides diagnostic functionality.

### 9. Kaggle Submission

`src/submission.py` handles submission-related functionality. The generated submission must contain the competition's identifier column, `Id`, and the predicted target column, `SalePrice`.

## Results

The following results were recorded during development.

| Evaluation | RMSLE |
|---|---:|
| Baseline validation | 0.138690 |
| Tuned Gradient Boosting validation | 0.135991 |
| Best recorded cross-validation result | 0.124434 |
| Kaggle public leaderboard score | 0.12846 |

The tuned Gradient Boosting model improved the recorded validation RMSLE by approximately **1.95%** relative to the baseline.

Cross-validation, validation-set evaluation, and Kaggle leaderboard scores measure performance in different ways. They should not be treated as directly interchangeable.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- SciPy
- Matplotlib and Seaborn
- PyYAML
- Kaggle API
- Jupyter Notebook

See `requirements.txt` for the project's pinned Python dependencies.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Naimul013/advanced_regression_housing_price.git
cd advanced_regression_housing_price
```

### 2. Create and activate a virtual environment

On Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Kaggle access

Configure your Kaggle API credentials according to the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

Ensure that your credentials are kept private and are never committed to the repository.

### 5. Configure the project

Review `config/config.yaml` and adjust the dataset, paths, experiment settings, and model parameters as required.

### 6. Run the workflow

The implementation is organized into separate Python modules for data loading, training, evaluation, cross-validation, tuning, finalization, and submission generation.

Run the appropriate workflow entry point for the experiment you want to execute. The exact execution sequence depends on the configuration and the functions exposed by each module.

## Reproducibility

The project uses centralized configuration and records experiment outputs to support reproducibility. Reproducing the exact results may also depend on the Python environment, dependency versions, random seeds, dataset access, and saved experiment artifacts.

## Limitations and Future Improvements

Potential areas for further improvement include:

- Strengthening preprocessing and feature-engineering tests.
- Adding automated tests for important pipeline components.
- Improving experiment tracking and artifact management.
- Validating dependency compatibility in a clean environment.
- Comparing performance across multiple validation strategies.
- Improving documentation of the end-to-end execution sequence.

## Author

**Naimul Islam**

GitHub: [@Naimul013](https://github.com/Naimul013)
