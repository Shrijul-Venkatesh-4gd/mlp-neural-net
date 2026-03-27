# Adult Dataset Preprocessing Steps

## Purpose

This document explains what preprocessing was applied to the Adult dataset before training the MLP, and why each step was used.

The implementation for these steps lives in [utils/data/preprocessing.py](/home/shrijul/Desktop/mlp-neural-net/utils/data/preprocessing.py).

## Pipeline Summary

The preprocessing pipeline does the following:

1. replace literal `?` values with real missing values
2. drop duplicate rows
3. drop `education`
4. drop `fnlwgt`
5. encode the target as `0` or `1`
6. split the data into train, validation, and test sets using stratification
7. fill missing categorical values with `Unknown`
8. apply `log1p` to `capital-gain` and `capital-loss`
9. one-hot encode categorical columns
10. standard-scale numeric columns
11. compute class weights from the training target
12. generate a 50-row random snapshot of the final processed data

## Step-By-Step Explanation

### 1. Replace `?` with missing values

Columns such as `workclass`, `occupation`, and `native-country` contain literal `?` values. These are not real categories, they are missing-like placeholders.

Why this matters:
- models should not treat `?` as a meaningful category
- missing values need to be handled consistently
- the EDA showed these columns have both `NaN` and `?`

### 2. Drop duplicate rows

Exact duplicate rows are removed before modeling.

Why this matters:
- duplicates can overweight repeated patterns
- removing them helps avoid biased learning from repeated samples
- it keeps the training data cleaner

### 3. Drop `education`

The pipeline drops `education` and keeps `education-num`.

Why this matters:
- `education-num` is already an ordinal encoding of `education`
- keeping both can double-count the same information
- keeping the numeric form is simpler for a first MLP baseline

### 4. Drop `fnlwgt`

The pipeline removes `fnlwgt` in the first-pass model setup.

Why this matters:
- the EDA showed almost no direct linear relationship with the target
- it simplifies the baseline feature set
- it can be added back later if experiments show value

### 5. Encode the target

The income label is converted to:

- `0` for `<=50K`
- `1` for `>50K`

Why this matters:
- neural networks need numeric targets
- this is the standard setup for binary classification

### 6. Split into train, validation, and test sets

The data is split using stratified sampling.

Why this matters:
- stratification preserves the original class balance in each split
- validation data is needed for tuning and early stopping
- test data must stay untouched until final evaluation

Important:
- preprocessing steps that learn from data, like scaling and one-hot encoding, are fitted only on the training set to avoid data leakage

### 7. Fill missing categorical values with `Unknown`

Missing categorical values are imputed with a dedicated label: `Unknown`.

Why this matters:
- dropping rows would waste a meaningful amount of data
- MLPs need a complete numeric matrix after encoding
- `Unknown` lets the model learn whether missingness carries signal

### 8. Apply `log1p` to `capital-gain` and `capital-loss`

These two numeric columns are log-transformed before scaling.

Why this matters:
- both columns are heavily skewed
- most values are zero, but a few are very large
- `log1p` compresses extreme values while preserving zeros

### 9. One-hot encode categorical columns

Categorical features such as `workclass`, `occupation`, `race`, `sex`, and `native-country` are expanded into binary indicator columns.

Why this matters:
- an MLP cannot directly consume string categories
- one-hot encoding turns categories into numeric input
- the cardinality in this dataset is manageable for a baseline model

### 10. Standard-scale numeric columns

Numeric columns are standardized after transformation.

Why this matters:
- MLPs train better when numeric features are on comparable scales
- columns like `capital-gain` and `education-num` originally have very different ranges
- scaling improves optimization stability

### 11. Compute class weights

Class weights are computed from the training labels.

Why this matters:
- the Adult dataset is imbalanced, with many more `<=50K` rows than `>50K`
- class weights help the model pay more attention to the minority class
- this is useful when training with weighted binary classification loss

### 12. Generate a 50-row snapshot

The pipeline creates a random 50-row sample from the final processed train, validation, and test matrices.

Why this matters:
- it gives a human-readable preview of the final model input
- it helps verify that scaling, encoding, and target formatting worked
- it makes debugging easier without printing the full dataset

## Final Output

After preprocessing, the data returned by the loader contains:

- `X_train`, `X_val`, `X_test`: final numeric feature matrices
- `y_train`, `y_val`, `y_test`: numeric binary labels
- `feature_names`: the generated processed column names
- `class_weights`: weights for imbalanced training
- `snapshot`: 50 random sampled rows from the final processed data

## Why This Is A Good First MLP Baseline

This preprocessing setup is a strong first baseline because it:

- keeps the pipeline simple
- handles missingness explicitly
- avoids obvious feature redundancy
- makes all inputs numeric
- scales the inputs for stable neural-network training
- preserves a proper evaluation split
- addresses class imbalance

Later improvements could include:

- testing `education` instead of `education-num`
- adding binary flags for whether `capital-gain` or `capital-loss` is non-zero
- comparing with tree-based models
- testing whether `fnlwgt` helps after the baseline is established
