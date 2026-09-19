import matplotlib
matplotlib.use('Agg')  # no windows

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import scatterplot, heatmap
from sklearn.tree import plot_tree


def show():
    plt.close('all')


def f_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc, model_name, thresholds=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle='--', label="Random classifier")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC plot. " + model_name)
    ax.legend()
    ax.grid()
    if thresholds is not None:
        for i in range(len(fpr)):
            ax.text(fpr[i], tpr[i], f"{thresholds[i]:.2f}")
    plt.savefig("data_out/plots/ROC_" + model_name + ".png", bbox_inches='tight', dpi=150)
    plt.close()


def f_roc_multiclass(fpr, tpr, roc_auc, classes, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    q = len(classes)
    for i in range(q):
        ax.plot(fpr[i], tpr[i],
                label=f"Class {i} ({classes[i]}) - AUC = {roc_auc[i]:.3f}")
    ax.plot(fpr["micro"], tpr["micro"], linestyle="--",
            label=f"micro-average - AUC = {roc_auc['micro']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves. " + model_name)
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.savefig("data_out/plots/ROC_" + model_name + ".png", bbox_inches='tight', dpi=150)
    plt.close()


def f_gain(df: pd.DataFrame, target_value, model_name):
    x = [0] + df["population"].tolist()
    y = [0] + df["gain"].tolist()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    ax.plot(x, y, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_xlabel("Fraction of population")
    ax.set_ylabel("Cumulative gain")
    ax.set_title("Gain chart - " + str(target_value) + ". " + model_name)
    ax.legend()
    ax.grid(True)
    plt.savefig("data_out/plots/Gain_" + str(target_value) + "_" + model_name + ".png",
                bbox_inches='tight', dpi=150)
    plt.close()


def f_lift(df: pd.DataFrame, target_value, model_name):
    x = [0] + df["population"].tolist()
    y = [1] + df["lift"].tolist()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, marker="o", label="Model")
    ax.axhline(1, linestyle="--", label="Random")
    ax.set_xlabel("Fraction of population")
    ax.set_ylabel("Lift")
    ax.set_title("Lift chart - " + str(target_value) + ". " + model_name)
    ax.legend()
    ax.grid(True)
    plt.savefig("data_out/plots/Lift_" + str(target_value) + "_" + model_name + ".png",
                bbox_inches='tight', dpi=150)
    plt.close()


def plot_conf_m(cm, classes, model_name):
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    cm_s = (cm.T / row_sums).T
    cm_p = cm / col_sums
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Normalized confusion matrix. " + model_name)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Row normalization (sensitivity)")
    cm_table1 = pd.DataFrame(cm_s, classes, classes)
    heatmap(cm_table1, vmin=0, vmax=1, cmap="Reds", annot=True, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Column normalization (precision)")
    cm_table2 = pd.DataFrame(cm_p, classes, classes)
    heatmap(cm_table2, vmin=0, vmax=1, cmap="Blues", annot=True, ax=ax2)
    plt.savefig("data_out/plots/CM_" + model_name + ".png", bbox_inches='tight', dpi=150)
    plt.close()


def plot_fi_dt(fi, model_name):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(fi["Predictors"], fi["Importance"])
    ax.invert_yaxis()
    ax.set_title("Feature importance. " + model_name)
    ax.set_xlabel("Importance")
    plt.savefig("data_out/plots/FI_DT_" + model_name + ".png", bbox_inches='tight', dpi=150)
    plt.close()


def plot_decision_tree(dt, predictors, classes):
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(1, 1, 1)
    class_labels = [str(c) for c in classes]
    # used max_depth=3 for readability
    plot_tree(dt, feature_names=predictors, class_names=class_labels,
              filled=True, rounded=True, fontsize=10, ax=ax, max_depth=3)
    ax.set_title("Decision tree (first 3 levels)", pad=35,
                 fontdict={"color": "b", "fontsize": 16})
    plt.savefig("data_out/plots/DTree.png", bbox_inches='tight', dpi=150)
    plt.close()


def plot_fi_rf(importance_df, model_name):
    x = np.arange(len(importance_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, importance_df["MDI"], width=width,
           label="Impurity reduction (MDI)")
    ax.bar(x + width / 2, importance_df["Permutation"], width=width,
           label="Permutation importance")
    ax.set_xticks(x)
    ax.set_xticklabels(importance_df["Predictors"], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature importance - Random Forest")
    ax.legend()
    fig.tight_layout()
    plt.savefig("data_out/plots/FI_RF_" + model_name + ".png", bbox_inches='tight', dpi=150)
    plt.close()


def f_scatter_classification(t: pd.DataFrame, y, model_name, suffix,
                              varx="Z1", vary="Z2", title="Instance plot"):
    f = plt.figure(title, figsize=(9, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontdict={"fontsize": 16})
    classes = np.unique(y)
    scatterplot(t, x=varx, y=vary, hue=y, style=y,
                hue_order=classes, style_order=classes, ax=ax, s=100)
    plt.savefig("data_out/plots/Plot_" + model_name + "_" + suffix + ".png", bbox_inches='tight', dpi=150)
    plt.close()
