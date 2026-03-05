# Device Status Prediction – Kaggle Submission


### For Reviewers

To check the project approach, please check the notebook:

`ml_arena_final_submission.ipynb`


## Overview

This project predicts whether a monitoring device is **working (0)** or **not working (1)** based on **47 numeric sensor features**.
The goal is to build a reliable classification model that can generalize well to unseen test data.

The approach follows common **Kaggle tabular competition practices**, including feature engineering, cross-validation, and model ensembling.

---

## Dataset

The dataset contains:

* **43k+ training samples**
* **47 numeric features (F01 – F47)**
* **Target column:** `Class`

  * `0` → device working
  * `1` → device not working

There are no missing values in the original dataset, but feature engineering can introduce NaNs which are handled during preprocessing.

---

## Exploratory Data Analysis

Key observations from EDA:

* Target distribution is moderately balanced (~60% working, ~40% not working).
* Some features are highly correlated (e.g., F01–F29, F02–F22).
* Several features are strongly skewed with large outliers.
* Individual features show moderate predictive power (AUC ~0.75).

These patterns suggest the dataset likely represents **sensor signals or frequency bands**, which benefits from interaction features.

---

## Feature Engineering

Several additional features were created to improve model performance.

### 1. Interaction Features

For correlated feature pairs, the following were generated:

* difference
* ratio
* sum

Example:

```
F01_F29_diff
F01_F29_ratio
F01_F29_sum
```

### 2. Row Statistics

Statistical summaries were computed across all features:

* mean
* standard deviation
* maximum
* minimum
* range

These help capture **global anomalies in sensor readings**.

### 3. Rank-Gauss Transformation

A **QuantileTransformer** was used to normalize feature distributions.
This technique is commonly used in Kaggle competitions to stabilize model training.

### 4. PCA Features

Principal Component Analysis (PCA) was applied to generate **10 additional components** that capture global variance in the dataset.

---

## Model Training

Three tree-based models were trained:

* **LightGBM**
* **XGBoost**
* **CatBoost**

These models are widely used in tabular machine learning competitions due to their strong performance.

### Cross-Validation

Training used **5-fold Stratified K-Fold cross-validation** to ensure stable performance and avoid overfitting.

Validation metric:

```
ROC-AUC
```

---

## Ensemble Strategy

Final predictions were created using a weighted average of the three models:

```
0.4 × LightGBM
0.4 × XGBoost
0.2 × CatBoost
```

Ensembling improves robustness and reduces model variance.

---

## Leakage and Overfitting Checks

Several checks were performed:

* Duplicate row detection
* Label shuffling test (AUC ≈ 0.50)
* Single-feature model tests

These confirmed that **no data leakage is present** and the model learns genuine patterns.

---


Format:

| Class       |
| ----------- |
| probability |

Each value represents the predicted probability that the device is **not working**.

---

## Tools Used

* Python
* Google Colab
* pandas / numpy
* scikit-learn
* LightGBM
* XGBoost
* CatBoost

---

## Conclusion

The pipeline combines **feature engineering, normalization, PCA, and model ensembling** to achieve strong predictive performance.
The approach is inspired by common strategies used in successful Kaggle tabular competition solutions.
