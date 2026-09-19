# Network Traffic Classification for Intrusion Detection

A supervised classification study that compares 11 machine learning models on the task of distinguishing normal network traffic from intrusions and anomalies, with an interactive web interface built on top of the trained models.

Dataset: [Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) (Kaggle).

## Overview

Each network connection is described by 41 attributes (protocol type, service, error rates, host activity counters, etc.) and labeled as either `normal` or `anomaly`. The project runs the standard stages of a classification study:

1. Predictor evaluation using five independent criteria: mutual information, the Fisher F test, the Chi squared test, the mRMR criterion and information value (IV/WOE).
2. Training and evaluation of 11 classifiers on a stratified 70/30 train/test split: Gaussian Naive Bayes, Kernel Density Naive Bayes, Linear Discriminant Analysis, a Decision Tree, Random Forest, Bagging, AdaBoost, a linear SVM, a Gaussian SVM, k-Nearest Neighbors and Logistic Regression.
3. Model selection by Matthews correlation coefficient (MCC), which accounts for all four cells of the confusion matrix and is not distorted by class imbalance.
4. Application of the optimal model to an unlabeled apply set.

Random Forest is the strongest model, with an MCC of 0.994 and 99.68 percent global accuracy on the held out test set. `src_bytes`, `dst_bytes` and `same_srv_rate` are the most influential predictors.

## Web interface

`app.py` is a Streamlit application that trains the full pipeline once (cached in memory) and exposes it through 6 sections:

- **Overview**: problem statement, dataset summary, class balance, methodology.
- **Predictor analysis**: ranked predictor scores across all five filtering criteria.
- **Model comparison**: a sortable table and chart comparing all 11 models by accuracy, CK index and MCC.
- **Model details**: per-model confusion matrix, ROC curve, gain and lift charts, feature importance and a 2D projection of the test set.
- **Live prediction**: a form for the most influential predictors that runs a live prediction through the optimal model, with every other predictor filled in from its typical training value.
- **Conclusions**: a summary of findings and limitations.

## Running locally

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the classification pipeline from the command line:

```bash
python main.py
```

This regenerates `data_out/tables` and `data_out/plots`.

Launch the web interface:

```bash
streamlit run app.py
```

The first load trains all 11 models (about a minute) and caches the result in memory for every subsequent visitor, so later navigation is instant.

## Tech stack

Python, pandas, NumPy, scikit-learn, mrmr-selection, matplotlib, seaborn, Streamlit, Plotly.

## TODO
A report about the results of this analysis will be added soon.
