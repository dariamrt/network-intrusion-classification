import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, roc_curve,
                             auc, roc_auc_score, matthews_corrcoef)
from sklearn.preprocessing import label_binarize


def calculate_metrics(y, prediction, classes):
    cm = confusion_matrix(y, prediction, labels=classes)
    cm_table = pd.DataFrame(cm, classes, classes)
    cm_table["Accuracy"] = np.diag(cm) * 100 / np.sum(cm, axis=1)
    global_accuracy = sum(np.diag(cm)) * 100 / len(y)
    mean_accuracy = cm_table["Accuracy"].mean()
    ck_index = cohen_kappa_score(y, prediction, labels=classes)
    mcc = matthews_corrcoef(y, prediction)
    accuracy = pd.Series(
        [global_accuracy, mean_accuracy, ck_index, mcc],
        ["Global accuracy", "Mean accuracy", "CK index", "MCC"],
        name="Accuracy indicators"
    )
    return cm_table, accuracy


def calculate_roc(y_test, y_score, classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    q = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    for i in range(q):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    auc_ovr_macro = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
    auc_ovr_weighted = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
    return fpr, tpr, roc_auc, auc_ovr_macro, auc_ovr_weighted


def calculate_gain_lift(y_true, y_prob, positive_target=None):
    if positive_target is None:
        negative_target = 0
    else:
        negative_target = "Non" + str(positive_target)

    y_num = (np.array(y_true) == positive_target).astype(int) if positive_target is not None else np.array(y_true, dtype=int)
    df = pd.DataFrame({"y_true": y_num, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    N = len(df)
    P = df["y_true"].sum()
    df["rank"] = np.arange(1, N + 1)
    df["cum_positive"] = df["y_true"].cumsum()
    df["population"] = df["rank"] / N
    df["decile"] = np.ceil(df["population"] * 10).astype(int)
    decile_table = df.groupby("decile").agg(
        population=("population", "max"),
        cum_positive=("cum_positive", "max")
    ).reset_index()
    decile_table["gain"] = decile_table["cum_positive"] / P
    decile_table["lift"] = decile_table["gain"] / decile_table["population"]
    return decile_table
