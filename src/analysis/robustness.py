import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import APPLY_FILE, DATA_OUT, TRAIN_FILE
from src.pipeline import CATEGORICAL_PREDICTORS, TARGET

OUTPUT = DATA_OUT / "robustness"
CATEGORICAL = CATEGORICAL_PREDICTORS.tolist()
POSITIVE = "anomaly"
N_FAMILIES = 8
MIN_FAMILY_SIZE = 50


def load_data():
    pd.set_option("future.infer_string", False)
    df = pd.read_csv(TRAIN_FILE)
    df_apply = pd.read_csv(APPLY_FILE)
    predictors = [c for c in df.columns if c != TARGET]
    return df, df_apply, predictors


def encode(x_train, x_test):
    categorical = [c for c in CATEGORICAL if c in x_train.columns]
    x_train = x_train.copy()
    x_test = x_test.copy()
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x_train[categorical] = encoder.fit_transform(x_train[categorical])
        x_test[categorical] = encoder.transform(x_test[categorical])
    return x_train, x_test


def random_forest():
    return RandomForestClassifier(random_state=0, n_jobs=-1)


def fit_score(model, x_train, x_test, y_train, y_test):
    x_tr, x_te = encode(x_train, x_test)
    model.fit(x_tr, y_train)
    prediction = model.predict(x_te)
    return model, prediction, matthews_corrcoef(y_test, prediction), accuracy_score(y_test, prediction)


def check_duplicates(df, df_apply, predictors, x_train, x_test):
    duplicated_vectors = df.duplicated(predictors)
    labels_per_vector = df.groupby(predictors)[TARGET].nunique()
    train_vectors = set(map(tuple, x_train.values))
    test_in_train = sum(tuple(r) in train_vectors for r in x_test.values)
    apply_in_train = len(df_apply.merge(df[predictors].drop_duplicates(), on=predictors))
    return pd.DataFrame({
        "Check": [
            "Exact duplicate rows",
            "Duplicate feature vectors",
            "Feature vectors with conflicting labels",
            "Test rows with an identical twin in the training split",
            "Apply set rows already present in the training data",
        ],
        "Value": [
            int(df.duplicated().sum()),
            int(duplicated_vectors.sum()),
            int((labels_per_vector > 1).sum()),
            int(test_in_train),
            apply_in_train,
        ],
    })


def cross_validation(df, predictors):
    preprocess = ColumnTransformer(
        [("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scoring = {"MCC": make_scorer(matthews_corrcoef), "Accuracy": "accuracy"}
    rows = []
    for name, model in [("Random Forest", random_forest()),
                        ("Decision tree, depth 3", DecisionTreeClassifier(max_depth=3, random_state=0))]:
        scores = cross_validate(Pipeline([("preprocess", preprocess), ("model", model)]),
                                df[predictors], df[TARGET], cv=cv, scoring=scoring)
        for metric in scoring:
            values = scores["test_" + metric]
            half_width = 2.145 * values.std(ddof=1) / np.sqrt(len(values))
            rows.append({"Model": name, "Metric": metric, "Mean": values.mean(),
                         "Std": values.std(ddof=1), "CI low": values.mean() - half_width,
                         "CI high": values.mean() + half_width})
    return pd.DataFrame(rows)


def shallow_trees(x_train, x_test, y_train, y_test):
    majority = pd.Series(y_train).mode().iloc[0]
    rows = [{"Model": "Always predict '" + majority + "'", "Splits": 0,
             "Accuracy": accuracy_score(y_test, np.full(len(y_test), majority)), "MCC": 0.0}]
    rules = ""
    for depth in [1, 2, 3]:
        model, _, mcc, acc = fit_score(DecisionTreeClassifier(max_depth=depth, random_state=0),
                                       x_train, x_test, y_train, y_test)
        rows.append({"Model": f"Decision tree, depth {depth}", "Splits": model.tree_.node_count // 2,
                     "Accuracy": acc, "MCC": mcc})
        if depth == 2:
            rules = export_text(model, feature_names=list(x_train.columns), decimals=2)
    model, _, mcc, acc = fit_score(random_forest(), x_train, x_test, y_train, y_test)
    rows.append({"Model": "Random Forest (all 41 predictors)", "Splits": None, "Accuracy": acc, "MCC": mcc})
    return pd.DataFrame(rows), rules, model


def single_predictor_power(x_train, x_test, y_train, y_test):
    rows = []
    for p in x_train.columns:
        _, _, mcc, acc = fit_score(DecisionTreeClassifier(max_depth=4, random_state=0),
                                   x_train[[p]], x_test[[p]], y_train, y_test)
        rows.append({"Predictor": p, "MCC": mcc, "Accuracy": acc})
    return pd.DataFrame(rows).sort_values("MCC", ascending=False).reset_index(drop=True)


def learning_curve(x_train, x_test, y_train, y_test):
    rows = []
    rng = np.random.RandomState(0)
    for fraction in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
        n = int(len(x_train) * fraction)
        idx = rng.choice(len(x_train), n, replace=False)
        _, _, mcc, acc = fit_score(random_forest(), x_train.iloc[idx], x_test,
                                   y_train[idx], y_test)
        rows.append({"Training rows": n, "Fraction": fraction, "MCC": mcc, "Accuracy": acc})
    return pd.DataFrame(rows)


def ablation(rf, x_train, x_test, y_train, y_test):
    ranking = x_train.columns[np.argsort(-rf.feature_importances_)]
    rows = []
    for k in [0, 1, 3, 5, 10, 15, 20, 25, 30]:
        keep = [c for c in x_train.columns if c not in ranking[:k]]
        _, _, mcc, acc = fit_score(random_forest(), x_train[keep], x_test[keep], y_train, y_test)
        rows.append({"Top predictors removed": k, "Removed": ", ".join(ranking[:k]),
                     "Predictors left": len(keep), "MCC": mcc, "Accuracy": acc})
    return pd.DataFrame(rows)


def attack_families(df, predictors):
    anomalies = df[df[TARGET] == POSITIVE]
    numeric = [p for p in predictors if p not in CATEGORICAL]
    features = pd.concat([np.log1p(anomalies[numeric].clip(lower=0)),
                          pd.get_dummies(anomalies[CATEGORICAL], dtype=float)], axis=1)
    z = StandardScaler().fit_transform(features)
    labels = KMeans(N_FAMILIES, random_state=0, n_init=10).fit_predict(z)
    return pd.Series(labels, index=anomalies.index, name="Family")


def describe_family(rows):
    pattern = rows[["protocol_type", "service", "flag"]].value_counts(normalize=True)
    (protocol, service, flag), share = pattern.index[0], pattern.iloc[0]
    return f"{protocol} / {service} / {flag}", share


def unseen_attacks(df, predictors, families, index_test, test_prediction):
    normal = df.index[df[TARGET] != POSITIVE]
    normal_train, normal_test = train_test_split(normal, test_size=0.3, random_state=0)
    seen = pd.Series(test_prediction, index=index_test)
    rows = []
    for family in sorted(families.unique()):
        held_out = families.index[families == family]
        if len(held_out) < MIN_FAMILY_SIZE:
            continue
        train_idx = normal_train.append(families.index[families != family])
        x_tr, x_te = encode(df.loc[train_idx, predictors],
                            df.loc[held_out.append(normal_test), predictors])
        model = random_forest().fit(x_tr, df.loc[train_idx, TARGET])
        prediction = pd.Series(model.predict(x_te), index=x_te.index)
        pattern, share = describe_family(df.loc[held_out])
        seen_rows = seen.loc[seen.index.intersection(held_out)]
        rows.append({
            "Family": int(family) + 1,
            "Connections": len(held_out),
            "Dominant pattern (protocol / service / flag)": pattern,
            "Pattern share": share,
            "Recall when seen in training": (seen_rows == POSITIVE).mean(),
            "Recall when unseen": (prediction.loc[held_out] == POSITIVE).mean(),
            "False positive rate on normal traffic": (prediction.loc[normal_test] == POSITIVE).mean(),
        })
    return pd.DataFrame(rows).sort_values("Connections", ascending=False).reset_index(drop=True)


def run_robustness(verbose=True):
    def log(message):
        if verbose:
            print(message)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    df, df_apply, predictors = load_data()
    x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(
        df[predictors], df[TARGET].values, df.index,
        test_size=0.3, random_state=0, stratify=df[TARGET].values
    )

    log("Checking duplicates")
    duplicates = check_duplicates(df, df_apply, predictors, x_train, x_test)
    log("Repeated 5-fold cross-validation")
    cv = cross_validation(df, predictors)
    log("Shallow tree baselines")
    baselines, rules, rf = shallow_trees(x_train, x_test, y_train, y_test)
    rf_test_prediction = rf.predict(encode(x_train, x_test)[1])
    log("Single predictor power")
    single = single_predictor_power(x_train, x_test, y_train, y_test)
    log("Learning curve")
    curve = learning_curve(x_train, x_test, y_train, y_test)
    log("Ablation of top predictors")
    ablated = ablation(rf, x_train, x_test, y_train, y_test)
    log("Leave-one-attack-family-out")
    families = attack_families(df, predictors)
    unseen = unseen_attacks(df, predictors, families, index_test, rf_test_prediction)

    results = {
        "duplicates": duplicates,
        "cross_validation": cv,
        "baselines": baselines,
        "single_predictor": single,
        "learning_curve": curve,
        "ablation": ablated,
        "unseen_attacks": unseen,
    }
    for name, table in results.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    (OUTPUT / "depth2_rules.txt").write_text(rules)
    results["depth2_rules"] = rules

    if verbose:
        pd.set_option("display.width", 200)
        for name, table in results.items():
            print("\n" + name)
            print(table)
        caught = (unseen["Recall when unseen"] * unseen["Connections"]).sum() / unseen["Connections"].sum()
        print(f"\nAttacks from unseen families caught: {100 * caught:.1f}%")
    return results


if __name__ == "__main__":
    run_robustness()
