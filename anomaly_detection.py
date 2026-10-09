from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import cross_dataset as cd
import robustness as rb

OUTPUT = Path("data_out/anomaly_detection")
TARGET_FPR = 0.01
DETECTORS = ["Isolation Forest", "Local Outlier Factor", "Autoencoder"]
COMBINED = "Any detector"
LOF_MAX_ROWS = 20000
SEED = 0


def log_positive(x):
    return np.log1p(np.clip(x, 0, None))


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("numeric", make_pipeline(FunctionTransformer(log_positive), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


class Autoencoder:

    def fit(self, x):
        self.network = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=200,
                                    early_stopping=True, random_state=SEED).fit(x, x)
        return self

    def anomaly_score(self, x):
        return ((self.network.predict(x) - x) ** 2).mean(axis=1)


class Detector:

    def __init__(self, name):
        self.name = name

    def fit(self, x):
        match self.name:
            case "Isolation Forest":
                self.model = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(x)
            case "Local Outlier Factor":
                rows = np.random.RandomState(SEED).permutation(len(x))[:LOF_MAX_ROWS]
                self.model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(x[rows])
            case "Autoencoder":
                self.model = Autoencoder().fit(x)
        return self

    def score(self, x):
        if self.name == "Autoencoder":
            return self.model.anomaly_score(x)
        return -self.model.score_samples(x)


def fit_detectors(x_fit, x_calibration):
    detectors = {}
    for name in DETECTORS:
        detector = Detector(name).fit(x_fit)
        calibration = detector.score(x_calibration)
        detector.threshold = np.quantile(calibration, 1 - TARGET_FPR)
        detector.strict_threshold = np.quantile(calibration, 1 - TARGET_FPR / len(DETECTORS))
        detectors[name] = detector
    return detectors


def alarms(detectors, x):
    flags, scores = {}, {}
    for name, detector in detectors.items():
        scores[name] = detector.score(x)
        flags[name] = scores[name] > detector.threshold
    flags[COMBINED] = np.any([scores[n] > detectors[n].strict_threshold for n in DETECTORS], axis=0)
    return flags, scores


def roc_points(y, score, method, n_points=200):
    fpr, tpr, _ = roc_curve(y, score)
    keep = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return pd.DataFrame({"Method": method, "False positive rate": fpr[keep], "Attacks detected": tpr[keep]})


def unseen_families():
    df, _, predictors = rb.load_data()
    families = rb.attack_families(df, predictors)
    normal = df.index[df[rb.TARGET] != rb.POSITIVE]
    normal_train, normal_test = train_test_split(normal, test_size=0.3, random_state=0)
    normal_fit, normal_calibration = train_test_split(normal_train, test_size=0.15, random_state=SEED)

    numeric = [p for p in predictors if p not in rb.CATEGORICAL]
    prep = preprocessor(numeric, rb.CATEGORICAL).fit(df.loc[normal_fit, predictors])
    x = lambda idx: prep.transform(df.loc[idx, predictors])
    detectors = fit_detectors(x(normal_fit), x(normal_calibration))

    anomalies = families.index
    flags_attack, scores_attack = alarms(detectors, x(anomalies))
    flags_normal, scores_normal = alarms(detectors, x(normal_test))
    y_roc = np.r_[np.zeros(len(normal_test)), np.ones(len(anomalies))]
    roc = pd.concat([roc_points(y_roc, np.r_[scores_normal[n], scores_attack[n]], n) for n in DETECTORS])

    rows = []
    rf_false_positives = []
    for family in sorted(families.unique()):
        held_out = families.index[families == family]
        if len(held_out) < rb.MIN_FAMILY_SIZE:
            continue
        in_family = families.loc[anomalies].values == family
        pattern, _ = rb.describe_family(df.loc[held_out])

        train_idx = normal_train.append(families.index[families != family])
        x_tr, x_te = rb.encode(df.loc[train_idx, predictors],
                               df.loc[held_out.append(normal_test), predictors])
        rf = rb.random_forest().fit(x_tr, df.loc[train_idx, rb.TARGET])
        rf_flags = pd.Series(rf.predict(x_te) == rb.POSITIVE, index=x_te.index)
        rf_false_positives.append(rf_flags.loc[normal_test].values)

        row = {"Family": int(family) + 1, "Connections": len(held_out),
               "Dominant pattern (protocol / service / flag)": pattern,
               "Random Forest": rf_flags.loc[held_out].mean()}
        for name in DETECTORS + [COMBINED]:
            row[name] = flags_attack[name][in_family].mean()
        row["Random Forest + any detector"] = (rf_flags.loc[held_out].values
                                               | flags_attack[COMBINED][in_family]).mean()
        rows.append(row)
    per_family = pd.DataFrame(rows).sort_values("Connections", ascending=False).reset_index(drop=True)

    methods = ["Random Forest"] + DETECTORS + [COMBINED, "Random Forest + any detector"]
    rf_fp = np.mean(rf_false_positives, axis=0)
    false_positive_rate = {name: flags_normal[name].mean() for name in DETECTORS + [COMBINED]}
    false_positive_rate["Random Forest"] = rf_fp.mean()
    false_positive_rate["Random Forest + any detector"] = np.mean(
        [(fp | flags_normal[COMBINED]).mean() for fp in rf_false_positives])
    weights = per_family["Connections"]
    summary = pd.DataFrame([{
        "Method": m,
        "Needs attack labels": m.startswith("Random Forest"),
        "Attacks detected": (per_family[m] * weights).sum() / weights.sum(),
        "Average over families": per_family[m].mean(),
        "Families above 50%": int((per_family[m] > 0.5).sum()),
        "False positive rate": false_positive_rate[m],
        "AUC": (roc_auc_score(y_roc, np.r_[scores_normal[m], scores_attack[m]])
                if m in DETECTORS else np.nan),
    } for m in methods])
    return summary, per_family, roc


def load_unsw_full(name):
    pd.set_option("future.infer_string", False)
    cd.download_unsw()
    df = pd.read_csv(cd.UNSW_FOLDER / name, encoding="utf-8-sig")
    df = df[df["proto"].isin(cd.PROTOCOLS)].reset_index(drop=True)
    df["attack_cat"] = df["attack_cat"].str.strip()
    return df


def new_network():
    train = load_unsw_full("UNSW_NB15_training-set.csv")
    test = load_unsw_full("UNSW_NB15_testing-set.csv")
    categorical = ["proto", "service", "state"]
    numeric = [c for c in train.columns if c not in categorical + ["id", "attack_cat", "label"]]
    y = test["label"].values

    normal = train[train["label"] == 0]
    normal_fit, normal_calibration = train_test_split(normal, test_size=0.15, random_state=SEED)
    prep = preprocessor(numeric, categorical).fit(normal_fit)
    detectors = fit_detectors(prep.transform(normal_fit), prep.transform(normal_calibration))
    flags, scores = alarms(detectors, prep.transform(test))

    predictions = {}
    kdd = cd.load_kdd()
    kdd_train, _ = train_test_split(kdd, test_size=0.3, random_state=0, stratify=kdd["label"])
    unsw_shared, _ = cd.load_unsw("UNSW_NB15_testing-set.csv")
    predictions["Random Forest trained on NSL-KDD"] = cd.train_predict(kdd_train, unsw_shared).astype(bool)

    kdd_normal = kdd[kdd["label"] == 0]
    kdd_fit, kdd_calibration = train_test_split(kdd_normal, test_size=0.15, random_state=SEED)
    shared_numeric = [f for f in cd.SHARED if f not in cd.CATEGORICAL]
    kdd_prep = preprocessor(shared_numeric, cd.CATEGORICAL).fit(kdd_fit[cd.SHARED])
    kdd_ae = Detector("Autoencoder").fit(kdd_prep.transform(kdd_fit[cd.SHARED]))
    kdd_threshold = np.quantile(kdd_ae.score(kdd_prep.transform(kdd_calibration[cd.SHARED])), 1 - TARGET_FPR)
    predictions["Autoencoder on NSL-KDD normal traffic"] = (
        kdd_ae.score(kdd_prep.transform(unsw_shared[cd.SHARED])) > kdd_threshold)

    for name in DETECTORS + [COMBINED]:
        predictions[name + " (local normal traffic)"] = flags[name]

    encoder = preprocessor([], categorical)
    x_train = np.c_[train[numeric].values, encoder.fit_transform(train)]
    x_test = np.c_[test[numeric].values, encoder.transform(test)]
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(x_train, train["label"])
    rf_name = "Random Forest with local attack labels"
    predictions[rf_name] = rf.predict(x_test).astype(bool)
    ranking = {name + " (local normal traffic)": scores[name] for name in DETECTORS}
    ranking[rf_name] = rf.predict_proba(x_test)[:, 1]
    roc = pd.concat([roc_points(y, s, m) for m, s in ranking.items()])

    needs = {"Random Forest trained on NSL-KDD": "NSL-KDD attack labels",
             "Autoencoder on NSL-KDD normal traffic": "Nothing from the new network",
             rf_name: "Labeled local attacks"}
    summary = pd.DataFrame([{
        "Method": m,
        "Needs from the new network": needs.get(m, "Local normal traffic only"),
        "Attacks detected": p[y == 1].mean(),
        "False positive rate": p[y == 0].mean(),
        "AUC": roc_auc_score(y, ranking[m]) if m in ranking else np.nan,
    } for m, p in predictions.items()])

    attacks = test["label"] == 1
    categories = pd.DataFrame({m: p[attacks] for m, p in predictions.items()})
    categories["Category"] = test.loc[attacks, "attack_cat"].values
    per_category = categories.groupby("Category").mean()
    per_category.insert(0, "Connections", categories.groupby("Category").size())
    per_category = per_category.sort_values("Connections", ascending=False).reset_index()
    return summary, per_category, roc


def run_anomaly_detection(verbose=True):
    def log(message):
        if verbose:
            print(message)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    log("NSL-KDD: attack families held out of training")
    nsl_summary, nsl_families, nsl_roc = unseen_families()
    log("UNSW-NB15: a new network without attack labels")
    unsw_summary, unsw_categories, unsw_roc = new_network()

    results = {
        "nsl_summary": nsl_summary,
        "nsl_families": nsl_families,
        "nsl_roc": nsl_roc,
        "unsw_summary": unsw_summary,
        "unsw_categories": unsw_categories,
        "unsw_roc": unsw_roc,
    }
    for name, table in results.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    if verbose:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 20)
        for name, table in results.items():
            if not name.endswith("roc"):
                print("\n" + name)
                print(table.round(4).to_string())
    return results


if __name__ == "__main__":
    run_anomaly_detection()
