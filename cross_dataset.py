import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OrdinalEncoder

OUTPUT = Path("data_out/cross_dataset")
UNSW_FOLDER = Path("data_in/unsw_nb15")
# the mirror swaps the official names: its train.csv is the official testing set
UNSW_FILES = {
    "UNSW_NB15_training-set.csv": (
        "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv",
        "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    ),
    "UNSW_NB15_testing-set.csv": (
        "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv",
        "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
    ),
}

SHARED = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
          "dst_host_count", "dst_host_srv_count"]
CATEGORICAL = ["protocol_type", "service", "flag"]
UNSW_COLUMNS = {"dur": "duration", "proto": "protocol_type", "service": "service",
                "state": "flag", "sbytes": "src_bytes", "dbytes": "dst_bytes",
                "ct_dst_ltm": "dst_host_count", "ct_srv_dst": "dst_host_srv_count"}
PROTOCOLS = ["tcp", "udp", "icmp"]
FLAG_MAP = {"FIN": "SF", "CON": "SF", "REQ": "S0", "RST": "RSTO"}
SERVICE_MAP = {"-": "other", "dns": "domain_u", "ftp-data": "ftp_data", "pop3": "pop_3",
               "irc": "IRC", "ssl": "http_443", "snmp": "other", "dhcp": "other",
               "radius": "other"}
SEED = 0


def download_unsw():
    UNSW_FOLDER.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in UNSW_FILES.items():
        path = UNSW_FOLDER / name
        if path.exists():
            continue
        print("Downloading", name)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256:
            tmp.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}")
        tmp.rename(path)


def load_kdd():
    pd.set_option("future.infer_string", False)
    df = pd.read_csv("data_in/Train_data.csv")
    out = df[SHARED].copy()
    out["label"] = (df["class"] == "anomaly").astype(int)
    out["attack_cat"] = np.where(out["label"] == 1, "Attack", "Normal")
    return out


def load_unsw(name):
    pd.set_option("future.infer_string", False)
    download_unsw()
    df = pd.read_csv(UNSW_FOLDER / name, encoding="utf-8-sig")
    kept = df[df["proto"].isin(PROTOCOLS)]
    out = kept[list(UNSW_COLUMNS)].rename(columns=UNSW_COLUMNS)
    out["service"] = out["service"].replace(SERVICE_MAP)
    out.loc[(out["service"] == "domain_u") & (out["protocol_type"] == "tcp"), "service"] = "domain"
    out["flag"] = np.where(out["protocol_type"] == "tcp",
                           out["flag"].map(FLAG_MAP).fillna("OTH"), "SF")
    out["label"] = kept["label"].values
    out["attack_cat"] = kept["attack_cat"].str.strip().values
    return out.reset_index(drop=True), len(df) - len(kept)


def encode(train, test):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_train = train[SHARED].copy()
    x_test = test[SHARED].copy()
    x_train[CATEGORICAL] = encoder.fit_transform(x_train[CATEGORICAL])
    x_test[CATEGORICAL] = encoder.transform(x_test[CATEGORICAL])
    return x_train, x_test


def random_forest():
    return RandomForestClassifier(random_state=SEED, n_jobs=-1)


def train_predict(train, test, features=None):
    features = features or SHARED
    x_train, x_test = encode(train, test)
    model = random_forest().fit(x_train[features], train["label"])
    return model.predict(x_test[features])


def rank_normalize(df):
    df = df.copy()
    for f in SHARED:
        if f not in CATEGORICAL:
            df[f] = df[f].rank(pct=True)
    return df


def scores(y, prediction):
    y = np.asarray(y)
    return {
        "MCC": matthews_corrcoef(y, prediction),
        "Accuracy": accuracy_score(y, prediction),
        "Attacks detected": prediction[y == 1].mean(),
        "False positive rate": prediction[y == 0].mean(),
    }


def transfer_matrix(kdd_train, kdd_test, unsw_train, unsw_test):
    rows = []
    for train_name, train, test_name, test in [
        ("NSL-KDD", kdd_train, "NSL-KDD", kdd_test),
        ("UNSW-NB15", unsw_train, "UNSW-NB15", unsw_test),
        ("NSL-KDD", kdd_train, "UNSW-NB15", unsw_test),
        ("UNSW-NB15", unsw_train, "NSL-KDD", kdd_test),
    ]:
        prediction = train_predict(train, test)
        rows.append({"Trained on": train_name, "Tested on": test_name,
                     "Same dataset": train_name == test_name,
                     **scores(test["label"], prediction)})
    return pd.DataFrame(rows)


def mapping_sensitivity(kdd_train, kdd_test, unsw_train, unsw_test):
    per_connection = [f for f in SHARED if f not in ("dst_host_count", "dst_host_srv_count")]
    variants = [
        ("All shared features", SHARED, False),
        ("Per-connection features only", per_connection, False),
        ("Rank-normalized within each dataset", SHARED, True),
    ]
    rows = []
    for name, features, normalize in variants:
        prepare = rank_normalize if normalize else (lambda d: d)
        for train_name, train, test_name, test in [("NSL-KDD", kdd_train, "UNSW-NB15", unsw_test),
                                                   ("UNSW-NB15", unsw_train, "NSL-KDD", kdd_test)]:
            prediction = train_predict(prepare(train), prepare(test), features)
            rows.append({"Variant": name, "Trained on": train_name, "Tested on": test_name,
                         **scores(test["label"], prediction)})
    return pd.DataFrame(rows)


def per_category(kdd_train, unsw_train, unsw_test):
    from_kdd = train_predict(kdd_train, unsw_test)
    from_unsw = train_predict(unsw_train, unsw_test)
    table = pd.DataFrame({"Category": unsw_test["attack_cat"],
                          "Trained on NSL-KDD": from_kdd,
                          "Trained on UNSW-NB15": from_unsw})
    grouped = table.groupby("Category")
    out = grouped.mean()
    out.insert(0, "Connections", grouped.size())
    out = out.reset_index().sort_values("Connections", ascending=False).reset_index(drop=True)
    return out


def adversarial_validation(kdd, unsw):
    n = min((kdd["label"] == 0).sum(), (unsw["label"] == 0).sum(), 10000)
    a = kdd[kdd["label"] == 0].sample(n, random_state=SEED)
    b = unsw[unsw["label"] == 0].sample(n, random_state=SEED)
    both = pd.concat([a, b], ignore_index=True)
    origin = np.r_[np.zeros(n), np.ones(n)]
    x = both[SHARED].copy()
    x[CATEGORICAL] = OrdinalEncoder().fit_transform(x[CATEGORICAL])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = [{"Features": "All shared features",
             "AUC": roc_auc_score(origin, cross_val_predict(random_forest(), x, origin, cv=cv,
                                                            method="predict_proba")[:, 1])}]
    for feature in SHARED:
        proba = cross_val_predict(random_forest(), x[[feature]], origin, cv=cv,
                                  method="predict_proba")[:, 1]
        rows.append({"Features": feature, "AUC": roc_auc_score(origin, proba)})
    return pd.DataFrame(rows)


def distribution_summary(kdd, unsw):
    numeric = [f for f in SHARED if f not in CATEGORICAL]
    rows = []
    for dataset, df in [("NSL-KDD", kdd), ("UNSW-NB15", unsw)]:
        for label, name in [(0, "Normal"), (1, "Attack")]:
            part = df[df["label"] == label]
            row = {"Dataset": dataset, "Class": name}
            row.update({f + " (median)": part[f].median() for f in numeric})
            rows.append(row)
    return pd.DataFrame(rows)


def adaptation_curve(kdd_train, unsw_train, unsw_test):
    rows = []
    for n in [0, 50, 200, 1000, 5000, 20000, len(unsw_train)]:
        local = (unsw_train.sample(n, random_state=SEED) if n < len(unsw_train) else unsw_train)
        for setting, train in [("NSL-KDD + local rows", pd.concat([kdd_train, local])),
                               ("Local rows only", local)]:
            if train["label"].nunique() < 2:
                continue
            prediction = train_predict(train, unsw_test)
            rows.append({"Labeled UNSW-NB15 rows": n, "Training data": setting,
                         **scores(unsw_test["label"], prediction)})
    return pd.DataFrame(rows)


def run_cross_dataset(verbose=True):
    def log(message):
        if verbose:
            print(message)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    kdd = load_kdd()
    kdd_train, kdd_test = train_test_split(kdd, test_size=0.3, random_state=0, stratify=kdd["label"])
    unsw_train, dropped_train = load_unsw("UNSW_NB15_training-set.csv")
    unsw_test, dropped_test = load_unsw("UNSW_NB15_testing-set.csv")

    datasets = pd.DataFrame([
        {"Dataset": "NSL-KDD (Kaggle)", "Recorded": "1998, simulated US Air Force LAN",
         "Rows used": len(kdd), "Attack share": kdd["label"].mean(), "Rows dropped": 0},
        {"Dataset": "UNSW-NB15 training set", "Recorded": "2015, UNSW Canberra cyber range",
         "Rows used": len(unsw_train), "Attack share": unsw_train["label"].mean(),
         "Rows dropped": dropped_train},
        {"Dataset": "UNSW-NB15 testing set", "Recorded": "2015, UNSW Canberra cyber range",
         "Rows used": len(unsw_test), "Attack share": unsw_test["label"].mean(),
         "Rows dropped": dropped_test},
    ])

    log("Train on one dataset, test on the other")
    transfer = transfer_matrix(kdd_train, kdd_test, unsw_train, unsw_test)
    log("Sensitivity to the feature mapping")
    sensitivity = mapping_sensitivity(kdd_train, kdd_test, unsw_train, unsw_test)
    log("Detection per UNSW-NB15 attack category")
    categories = per_category(kdd_train, unsw_train, unsw_test)
    log("Adversarial validation")
    adversarial = adversarial_validation(kdd, unsw_train)
    distributions = distribution_summary(kdd, unsw_train)
    log("Adaptation curve")
    adaptation = adaptation_curve(kdd_train, unsw_train, unsw_test)

    results = {
        "datasets": datasets,
        "transfer": transfer,
        "sensitivity": sensitivity,
        "categories": categories,
        "adversarial": adversarial,
        "distributions": distributions,
        "adaptation": adaptation,
    }
    for name, table in results.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    if verbose:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 20)
        for name, table in results.items():
            print("\n" + name)
            print(table.round(4).to_string())
    return results


if __name__ == "__main__":
    run_cross_dataset()
