# Yelp Hybrid Recommendation System

This repository contains the implementation of a hybrid recommendation system developed for the Yelp dataset, combining item-based collaborative filtering, model-based collaborative filtering (content-based), and a meta-learner approach. The system predicts user ratings for businesses based on historical interaction data and additional metadata provided by Yelp.

## Repository Structure

```
yelp-recommendation-system/
├── .venv/
├── evaluation/
│   └── eval.py
├── src/
│   └── modules/
│       ├── content_based_recommendation.py
│       ├── item_based_collaborative.py
│       └── hybrid_recommendation.py
├── tuning/
│   ├── linear_regression_experimentation.ipynb
│   └── model_based_tuning.ipynb
├── README.md
└── requirements.txt
```

## Setup

### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Ensure you have Java and Apache Spark installed on your system, as PySpark relies on these.

## Dataset

Download the Yelp dataset (user, business, review, tip, photo, and check-in data) from [Yelp Dataset Challenge](https://www.yelp.com/dataset). Place the dataset files within a directory named `data` in the root of this project.

```
data/
├── business.json
├── checkin.json
├── photo.json
├── review_train.json
├── tip.json
└── user.json
```

## Running the Recommendation System

The recommendation system is implemented using PySpark. Make sure Spark is correctly set up and configured.

### Generating Recommendations

Navigate to the source directory:

```bash
cd src/modules
```

Run the hybrid recommendation pipeline:

```bash
spark-submit hybrid_recommendation.py <path_to_data_directory> <path_to_test_csv> <output_predictions_csv>
```

- Replace `<path_to_data_directory>` with the location of your Yelp data folder.
- Replace `<path_to_test_csv>` with the CSV file containing user-business pairs for which you want predictions.
- Replace `<output_predictions_csv>` with the desired output file path for the predictions.

Example:

```bash
spark-submit hybrid_recommendation.py ../../data ../../data/yelp_test.csv ../../outputs/predictions.csv
```

## Evaluating Predictions

Use the evaluation script provided in the `evaluation` directory:

Navigate to evaluation:

```bash
cd ../../evaluation
```

Open `eval.py` and set the following variables to match your files:

```python
predictions_file = "path/to/your/predictions.csv"
ground_truth_file = "path/to/your/ground_truth.csv"
```

Run the evaluation script:

```bash
python eval.py
```

The script calculates the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Error Distribution

## Hyperparameter Tuning

Hyperparameter tuning for the model-based collaborative filtering method is available in Jupyter notebooks located in the `tuning` directory:

- `linear_regression_experimentation.ipynb`: Experiments using linear regression models.
- `model_based_tuning.ipynb`: Grid search and tuning of hyperparameters for the XGBoost model.

Run these notebooks using:

```bash
jupyter notebook
```

Navigate through the notebook to execute cells and analyze results.

## Contributions

Contributions, feature requests, and bug reports are welcome! Feel free to create an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
