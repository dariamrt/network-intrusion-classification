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

A near-perfect score on an intrusion dataset is a red flag, so `src/analysis/robustness.py` stress tests it:

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

## Does it work on another network?

`src/analysis/cross_dataset.py` evaluates the same kind of model on [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), a dataset recorded in 2015 on a different network with modern attack tools. The two datasets share 8 comparable attributes (duration, protocol, service, connection state, bytes in each direction and two host-level connection counters), so every model in this experiment uses only those 8.

| Trained on | Tested on | MCC | Attacks detected | False positive rate |
|---|---|---|---|---|
| NSL-KDD | NSL-KDD | 0.993 | 99.6% | 0.3% |
| UNSW-NB15 | UNSW-NB15 | 0.711 | 95.7% | 26.4% |
| **NSL-KDD** | **UNSW-NB15** | **-0.125** | **0.07%** | 3.2% |
| UNSW-NB15 | NSL-KDD | -0.239 | 17.7% | 39.5% |

- **Every modern attack category is missed.** The NSL-KDD model detects under 1% of Exploits, Fuzzers, Generic, DoS, Reconnaissance, Backdoor and Worms traffic.
- **The networks are completely different.** In adversarial validation, a classifier separates the two datasets' *normal* traffic with AUC 1.00, and almost any single numeric feature is enough.
- **The feature mapping is not to blame.** Dropping the connection counters or rank-normalizing every feature within its own dataset still leaves the NSL-KDD to UNSW-NB15 transfer below zero MCC. Interestingly, with rank normalization a model trained on the broader, modern UNSW-NB15 attack set *does* transfer back to NSL-KDD (MCC 0.75).
- **Local data beats public data.** 50 labeled UNSW-NB15 connections already lift MCC from -0.12 to 0.49. Adding the 25,000 NSL-KDD rows on top of local data changes MCC by at most about 0.015.

**Conclusion:** a benchmark score measures how well a model knows one environment. Deploying an intrusion detector requires labeled data and evaluation from the target network, plus periodic retraining.

UNSW-NB15 is downloaded automatically (about 47 MB, checksum verified) on the first run into `data_in/unsw_nb15`. It is the official training/testing partition, by N. Moustafa and J. Slay.

## Catching attacks it has never seen

Both experiments above expose the same weakness: a classifier trained on labeled attacks only recognizes those attacks. `src/analysis/anomaly_detection.py` tests the standard remedy. Detectors that learn only what *normal* traffic looks like (Isolation Forest, Local Outlier Factor and an autoencoder) flag anything that deviates, so they need no attack labels. Thresholds are set on held-out normal traffic to flag about 1% of it.

**NSL-KDD, attack families held out of training:**

| Method | Needs attack labels | Unseen attacks detected | False alarms |
|---|---|---|---|
| Random Forest | yes | 9.8% | 0.1% |
| Isolation Forest | no | 79.3% | 0.6% |
| Autoencoder | no | 59.7% | 1.1% |
| **Random Forest + anomaly detectors** | yes | **69.5%** | 0.7% |

**UNSW-NB15, a new network with no labeled attacks:**

| Method | Needs from the new network | Attacks detected | False alarms |
|---|---|---|---|
| Random Forest trained on NSL-KDD | nothing | 0.07% | 3.2% |
| Autoencoder trained on NSL-KDD normal traffic | nothing | 27.2% | 68.6% |
| **Autoencoder trained on local normal traffic** | normal traffic only | **65.7%** | 2.6% |
| Random Forest with every local attack labeled | labeled attacks | 98.3% | 27.4% |

- **Anomaly detection fills much of the gap.** It needs only normal traffic, which every network has plenty of.
- **Labels still help.** At the same 3% false alarm rate, the labeled Random Forest catches 84% of attacks and the autoencoder 67% (AUC 0.97 vs 0.92).
- **Normal-only training must be local too.** An autoencoder trained on NSL-KDD's normal traffic flags most of UNSW-NB15's normal traffic.
- **Some attacks look normal.** The ICMP and HTTP attack families stay hidden from every detector.
- **No detector wins everywhere.** Isolation Forest is best on NSL-KDD and the autoencoder on UNSW-NB15. Picking one requires a few labeled attacks to validate on, not the test results.

**Conclusion:** the practical design combines both. A classifier handles known attacks, and an anomaly detector trained on the target network's own traffic acts as a safety net for new ones.

## Web interface

`app.py` is a Streamlit application that loads the precomputed results and presents them in 9 sections:

- **Overview**: problem statement, dataset summary, class balance, methodology.
- **Predictor analysis**: ranked predictor scores across all five filtering criteria.
- **Model comparison**: a sortable table and chart comparing all 11 models by accuracy, CK index and MCC.
- **Model details**: per-model confusion matrix, ROC curve, gain and lift charts, feature importance and a 2D projection of the test set.
- **Is 99.7% too good to be true?**: the stress tests above, with charts for the shallow tree baselines, learning curve, predictor ablation and unseen attack families.
- **Does it work on another network?**: the NSL-KDD to UNSW-NB15 transfer experiments, attack categories, adversarial validation and the local data curve.
- **Catching attacks it has never seen**: anomaly detectors on the held-out attack families and on UNSW-NB15, including the trade-off between detected attacks and false alarms.
- **Live prediction**: a form for the most influential predictors that runs a live prediction through the optimal model, with every other predictor filled in from its typical training value.
- **Conclusions**: a summary of findings and limitations.

## Project structure

| Path | Contents |
|---|---|
| `app.py` | Streamlit interface |
| `main.py` | runs every analysis and saves the results the app loads |
| `src/config.py` | data and output paths |
| `src/pipeline.py` | the classification study: split, predictor scoring, 11 models, apply set |
| `src/classifiers.py` | model definitions and evaluation |
| `src/predictor_scoring.py` | mutual information, Fisher F, Chi squared, mRMR and IV/WOE |
| `src/metrics.py` | accuracy indicators, ROC, gain and lift |
| `src/kde_naive_bayes.py` | the kernel density Naive Bayes classifier |
| `src/plots.py` | matplotlib figures |
| `src/analysis/` | stress tests, cross-dataset evaluation and anomaly detection |
| `data_in/` | NSL-KDD training and apply sets (UNSW-NB15 is downloaded here on first run) |
| `data_out/` | generated tables, plots and the saved results |

## Running locally

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the web interface:

```bash
streamlit run app.py
```

The app loads precomputed results from `data_out`, so it starts in a few seconds without retraining anything.

Rerun every analysis and refresh those results (downloads UNSW-NB15 on the first run, then about two minutes):

```bash
python main.py
```

Each analysis can also be run on its own from the project folder:

| Command | What it does | Output |
|---|---|---|
| `python -m src.pipeline` | predictor scoring and the 11 classifiers | `data_out/tables`, `data_out/plots` |
| `python -m src.analysis.robustness` | stress tests of the headline result | `data_out/robustness` |
| `python -m src.analysis.cross_dataset` | NSL-KDD versus UNSW-NB15 | `data_out/cross_dataset` |
| `python -m src.analysis.anomaly_detection` | detectors trained only on normal traffic | `data_out/anomaly_detection` |

## Tech stack

Python, pandas, NumPy, scikit-learn, mrmr-selection, matplotlib, seaborn, Streamlit, Plotly.

## TODO
A report about the results of this analysis will be added soon.
