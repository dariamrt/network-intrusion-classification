import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from functions import calculate_metrics, calculate_roc, calculate_gain_lift, KDENaiveBayes
from plots import (f_roc_multiclass, f_lift, f_gain, show,
                   plot_fi_dt, plot_decision_tree, plot_conf_m,
                   f_roc, f_scatter_classification, plot_fi_rf)

# used a sample size for slow models on large datasets
SAMPLE_KDE_TRAIN = 3000
SAMPLE_KDE_TEST  = 2000
SAMPLE_SVM       = 5000

def _stratified_sample(x, y, n, random_state=0):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    idx, _ = next(sss.split(x, y))
    return x[idx], y[idx], idx


def evaluate_model(y_test, y, proba, q, classes, model_name):
    report = pd.DataFrame(classification_report(y_test, y, output_dict=True)).transpose()
    report.to_csv("data_out/tables/report_" + model_name + ".csv")

    cm_table, accuracy = calculate_metrics(y_test, y, classes)
    cm_table.to_csv("data_out/tables/cm_" + model_name + ".csv")
    accuracy.to_csv("data_out/tables/accuracy_" + model_name + ".csv")
    plot_conf_m(cm_table.values[:, :-1], classes, model_name)
    show()

    if q > 2:
        fpr, tpr, roc_auc, auc_ovr_macro, auc_ovr_weighted = calculate_roc(y_test, proba, classes)
        print("Average AUC:", auc_ovr_macro)
        print("Weighted average AUC:", auc_ovr_weighted)
        f_roc_multiclass(fpr, tpr, roc_auc, classes, model_name)
        for i in range(q):
            decile_df = calculate_gain_lift(y_test, proba[:, i], classes[i])
            decile_df.to_csv("data_out/tables/gain_lift_" + str(classes[i]) + "_" + model_name + ".csv")
            f_lift(decile_df, classes[i], model_name)
            f_gain(decile_df, classes[i], model_name)
        show()
    else:
        pos_label = classes[1]
        fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1], pos_label=pos_label)
        auc_value = roc_auc_score(y_test, proba[:, 1])
        f_roc(fpr, tpr, auc_value, model_name, thresholds if len(thresholds) < 10 else None)
        decile_df = calculate_gain_lift(y_test, proba[:, 1], classes[1])
        decile_df.to_csv("data_out/tables/gain_lift_" + str(classes[1]) + "_" + model_name + ".csv")
        f_lift(decile_df, classes[1], model_name)
        f_gain(decile_df, classes[1], model_name)
        show()

    return accuracy["MCC"]


def classify(x_train, x_test, y_train, y_test,
             test_predictions, model_name,
             predictors=None, model2d=None, scaling=None,
             df_train=None, df_test=None):
    q = len(np.unique(y_train))

    # indexing for models that use a sample
    idx_test_kde = None

    match model_name:
        case "GaussianNB":
            model = GaussianNB()
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "LDA":
            model = LinearDiscriminantAnalysis()
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "KDE":
            # KDE is O(n*d) at prediction time so I used a stratified sample
            x_tr_s, y_tr_s, _ = _stratified_sample(x_train, y_train, SAMPLE_KDE_TRAIN)
            x_te_s, y_te_s, idx_test_kde = _stratified_sample(x_test, y_test, SAMPLE_KDE_TEST)
            model = KDENaiveBayes()
            x_tr, x_te, y_tr, y_te = x_tr_s, x_te_s, y_tr_s, y_te_s
        case "Bagging":
            model = BaggingClassifier(random_state=0, n_jobs=-1)
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "RF":
            model = RandomForestClassifier(random_state=0, n_jobs=-1)
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "DT":
            model = DecisionTreeClassifier(random_state=0)
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "ADABoost":
            model = AdaBoostClassifier(random_state=0)
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case "SVMLin":
            # CalibratedClassifierCV because it has at least O(n^2) complexity
            model = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=0), n_jobs=-1)
            x_tr = scaling.transform(x_train)
            x_te = scaling.transform(x_test)
            y_tr, y_te = y_train, y_test
        case "SVM_Gaussian":
            # used a stratified sample
            x_tr_s, y_tr_s, _ = _stratified_sample(
                scaling.transform(x_train), y_train, SAMPLE_SVM
            )
            model = SVC(C=10, probability=True, random_state=0)
            x_tr = x_tr_s
            x_te = scaling.transform(x_test)
            y_tr, y_te = y_tr_s, y_test
        case "kNN":
            model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            x_tr = scaling.transform(x_train)
            x_te = scaling.transform(x_test)
            y_tr, y_te = y_train, y_test
        case "RegL":
            if q > 2:
                model = LogisticRegression(C=10, multi_class='ovr',
                                           max_iter=1000, random_state=0, n_jobs=-1)
            else:
                model = LogisticRegression(C=10, max_iter=1000, random_state=0)
            x_tr, x_te, y_tr, y_te = x_train, x_test, y_train, y_test
        case _:
            print("Model does not exist", model_name)
            return

    model.fit(x_tr, y_tr)
    y = model.predict(x_te)
    proba = model.predict_proba(x_te)
    classes = model.classes_

    # for KDE, test_predictions is updated only on the subsample
    if idx_test_kde is not None:
        test_predictions.loc[test_predictions.index[idx_test_kde], model_name] = y
    else:
        test_predictions[model_name] = y

    mcc = evaluate_model(y_te, y, proba, q, classes, model_name)

    # full error table with all predictors
    error_mask = y != y_te
    if idx_test_kde is not None:
        global_idx = test_predictions.index[idx_test_kde]
    else:
        global_idx = test_predictions.index

    if df_test is not None and error_mask.any():
        errors = df_test.loc[global_idx[error_mask]].copy()
        errors["Prediction"] = y[error_mask]
        errors["Actual"] = y_te[error_mask]
    else:
        errors = pd.DataFrame({
            "Actual": y_te[error_mask],
            "Prediction": y[error_mask]
        }, index=global_idx[error_mask])
    errors.to_csv("data_out/tables/errors_" + model_name + ".csv")

    match model_name:
        case "DT":
            fi = pd.DataFrame({
                "Predictors": predictors,
                "Importance": model.feature_importances_
            })
            plot_fi_dt(fi, model_name)
            fi.to_csv("data_out/tables/FI_DT.csv", index=False)
            plot_decision_tree(model, predictors, classes)
            show()
        case "RF":
            fi_table = pd.DataFrame({
                "Predictors": predictors,
                "MDI": model.feature_importances_
            })
            perm_importance = permutation_importance(
                model, x_te, y_te,
                n_repeats=10, random_state=42, scoring="accuracy", n_jobs=-1
            )
            fi_table["Permutation"] = perm_importance.importances_mean
            plot_fi_rf(fi_table, model_name)
            fi_table.to_csv("data_out/tables/FI_RF.csv", index=False)
            show()

    # 2D scatter for correctly/incorrectly classified instances
    if model2d is not None:
        x_test_2d = model2d.fit_transform(x_te)
        df_2d = pd.DataFrame(x_test_2d, columns=["Z1", "Z2"])
        scaling_name = model2d.__class__.__name__

        f_scatter_classification(df_2d, y_te, model_name, "classes",
                                 title="Class plot. Model:" + model_name +
                                       ". Scaling:" + scaling_name)
        f_scatter_classification(df_2d, y, model_name, "predictions",
                                 title="Prediction plot. Model:" + model_name +
                                       ". Scaling:" + scaling_name)
        err_y = np.empty(len(y), dtype=object)
        err_y[y == y_te] = "Correct"
        err_y[y != y_te] = "Incorrect"
        f_scatter_classification(df_2d, err_y, model_name, "errors",
                                 title="Misclassifications. Model:" + model_name +
                                       ". Scaling:" + scaling_name)
        show()

    return model, mcc
