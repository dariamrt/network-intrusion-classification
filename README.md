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

## Is 99.7% too good to be true?

A near-perfect score on an intrusion dataset is a red flag, so `robustness.py` stress tests it:

| Check | Result |
|---|---|
| Duplicates / train-test leakage | None. 0 test rows have an identical twin in training; encoders are fit on training data only. |
| Repeated 5-fold cross-validation (15 fits) | MCC 0.995 ± 0.001, so the single split was not lucky. |
| Decision tree with **one** split | 91.9% accuracy. Four `if` rules (depth 2) reach 94.8%. |
| Learning curve | 176 training rows (1% of the data) already give MCC 0.94. |
| Remove the 10 most important predictors | MCC still 0.97; the signal is highly redundant. |
| **Hold out an entire attack family** | **Only 9.8% of unseen attacks are detected**, versus about 99.7% for the same attacks when they are seen in training. |

The dataset has no attack type labels, so attack families are approximated by clustering the anomalies (KMeans on log-scaled, one-hot encoded predictors). Each family is then removed from training and used as the test set.

**Conclusion:** the 99.7% is not caused by leakage. It reflects an easy test design in which every attack type appears in both training and test. The model recognizes known attacks very well but has not learned what an attack looks like in general. A realistic deployment would pair it with anomaly detection trained on normal traffic and evaluate it on traffic from a different time or network.

## Web interface

`app.py` is a Streamlit application that trains the full pipeline once (cached in memory) and exposes it through 6 sections:

- **Overview**: problem statement, dataset summary, class balance, methodology.
- **Predictor analysis**: ranked predictor scores across all five filtering criteria.
- **Model comparison**: a sortable table and chart comparing all 11 models by accuracy, CK index and MCC.
- **Model details**: per-model confusion matrix, ROC curve, gain and lift charts, feature importance and a 2D projection of the test set.
- **Is 99.7% too good to be true?**: the stress tests above, with charts for the shallow tree baselines, learning curve, predictor ablation and unseen attack families.
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

Run the stress tests (about 15 seconds):

```bash
python robustness.py
```

This writes its tables to `data_out/robustness`.

Launch the web interface:

```bash
streamlit run app.py
```

The first load trains all 11 models (about a minute) and caches the result in memory for every subsequent visitor, so later navigation is instant.

## Tech stack

Python, pandas, NumPy, scikit-learn, mrmr-selection, matplotlib, seaborn, Streamlit, Plotly.

## TODO
A report about the results of this analysis will be added soon.
