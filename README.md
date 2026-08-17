# Forest Fire Classification Project

## Project objective

This project predicts whether a forest-fire observation is classified as `fire` or `not fire` using weather measurements, Fire Weather Index (FWI) components, region, and temporal information.

## Project structure

1. `01_data_audit_and_cleaning_final.ipynb`  
   Reconstructs and validates the raw CSV, repairs the known malformed row, standardizes types, and saves `forest_fires_clean.csv`.

2. `02_eda_and_feature_engineering_final.ipynb`  
   Creates the stratified train/test split, performs EDA on the training set only, engineers `DayOfYear`, and saves the engineered train/test CSV files.

3. `03_preprocessing_and_modeling_final.ipynb`  
   Defines model-specific preprocessing pipelines; develops Logistic Regression, LDA, QDA, KNN, tree-based models, and SVMs; performs cross-validation and hyperparameter tuning; and completes the final test-set comparison, interpretation, limitations, and conclusion.

## How to run

Place `forest_fires_dataset.csv` in the same directory as the notebooks and run the notebooks in numerical order: **01 → 02 → 03**.

Intermediate CSV files are generated automatically by Notebooks 01 and 02.

## Models evaluated

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- K-Nearest Neighbors (KNN)
- Unpruned and pruned Decision Trees
- Bagging
- Random Forest
- Gradient Boosting
- Support Vector Machines with linear, RBF, and polynomial kernels

## Evaluation approach

- Stratified 80/20 train/test split
- 5-fold stratified cross-validation for model development
- Model-specific preprocessing inside scikit-learn pipelines
- Hyperparameter selection for KNN, tree pruning, and SVM
- Final held-out test evaluation using accuracy, precision, recall/sensitivity, specificity, F1 score, confusion matrices, ROC curves, and AUC

## Main result

Several models perform very strongly. Bagging provides the strongest overall predictive profile, while the pruned decision tree is a particularly interpretable alternative. `FFMC`, `ISI`, and `FWI` consistently appear among the most influential predictors, although their strong correlations require cautious interpretation.
