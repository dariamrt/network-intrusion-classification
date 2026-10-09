import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classifiers import classify
from src.config import APPLY_FILE, TABLES, TRAIN_FILE, initialize_output_folders
from src.predictor_scoring import filtering

TARGET = "class"
CATEGORICAL_PREDICTORS = np.array(["protocol_type", "service", "flag"])
SCALED_MODELS = ["SVMLin", "SVM_Gaussian", "kNN"]

MODELS = [
    "GaussianNB",    # Bayesian parametric Gaussian
    "KDE",           # Bayesian non-parametric kernel
    "LDA",           # Linear classification
    "DT",            # Decision tree
    "RF",            # Random Forest
    "Bagging",       # Bagging
    "ADABoost",      # AdaBoost
    "SVMLin",        # Linear SVM
    "SVM_Gaussian",  # SVM Gaussian
    "kNN",           # kNN
    "RegL"           # Logistic regression
]


def run_pipeline(verbose=True):
    pd.set_option("display.max_columns", None)
    # pandas defaults string columns to Arrow backed arrays when pyarrow is
    # installed (a dependency of Streamlit); scikit-learn's fancy indexing
    # in train_test_split does not support that, so plain numpy object
    # arrays are restored here.
    pd.set_option("future.infer_string", False)

    df = pd.read_csv(TRAIN_FILE)
    df_apply = pd.read_csv(APPLY_FILE)

    target = TARGET
    predictors = np.array([c for c in df.columns if c != target])
    categorical_predictors = CATEGORICAL_PREDICTORS
    numeric_predictors = np.array([v for v in predictors if v not in categorical_predictors])

    y = df[target].values
    index_train, index_test = train_test_split(
        df.index, test_size=0.3, random_state=0, stratify=y
    )
    df_train = df.loc[index_train].copy()
    df_test_orig = df.loc[index_test].copy()
    y_train = df_train[target].values
    y_test = df_test_orig[target].values

    categorical_options = {c: sorted(df_train[c].unique().tolist()) for c in categorical_predictors}
    categorical_defaults = {c: df_train[c].mode().iloc[0] for c in categorical_predictors}
    numeric_defaults = df_train[numeric_predictors].median().to_dict()
    numeric_ranges = {
        c: (float(df_train[c].min()), float(df_train[c].max())) for c in numeric_predictors
    }

    initialize_output_folders()

    if verbose:
        print("Filtering predictors")
    # predictor scores, encoder and scaler only see the training split
    df_predictors, x_train, encoder = filtering(
        df_train, predictors, categorical_predictors, numeric_predictors, target
    )
    if verbose:
        print("Predictor order by standardized average score:")
        print(df_predictors)
    df_predictors.to_csv(TABLES / "Predictors.csv")

    df_test_encoded = df_test_orig.copy()
    df_test_encoded[categorical_predictors] = encoder.transform(df_test_orig[categorical_predictors])
    x_test = df_test_encoded[predictors].values

    # 2D reduction via PCA (large set of >5000 instances)
    model2d = PCA(n_components=2)

    # standardization for kNN, SVMLin, SVM_Gaussian
    scaling = StandardScaler()
    scaling.fit(x_train)

    # select optimal model based on MCC
    optimal_model = None
    optimal_model_name = None
    optimal_mcc = -1
    model_mcc = {}

    test_predictions = pd.DataFrame(index=index_test)
    test_predictions[target] = y_test

    for model_name in MODELS:
        if verbose:
            print("Running model", model_name)
        model, mcc = classify(
            x_train, x_test, y_train, y_test,
            test_predictions, model_name,
            predictors, model2d, scaling,
            df_train=df_train,
            df_test=df_test_orig
        )
        model_mcc[model_name] = mcc
        if mcc > optimal_mcc:
            optimal_mcc = mcc
            optimal_model = model
            optimal_model_name = model_name
        if verbose:
            print("Model", model_name, "executed. MCC =", round(mcc, 4))

    test_predictions.to_csv(TABLES / "Test_predictions.csv")
    if verbose:
        print("\nOptimal model:", optimal_model_name, "| MCC =", round(optimal_mcc, 4))

    # apply optimal model on the apply set
    if verbose:
        print("Applying model", optimal_model_name, "on the apply set")
    df_apply_ = df_apply.copy()

    if len(categorical_predictors) > 0:
        x_ = encoder.transform(df_apply_[categorical_predictors])
        for i in range(len(categorical_predictors)):
            df_apply_[categorical_predictors[i]] = x_[:, i]

    x_apply = df_apply_[predictors].values

    if optimal_model_name in SCALED_MODELS:
        y_predict = optimal_model.predict(scaling.transform(x_apply))
    else:
        y_predict = optimal_model.predict(x_apply)

    df_apply["Predict"] = y_predict
    df_apply.to_csv(TABLES / "Apply_predictions.csv")
    if verbose:
        print("Apply prediction saved in", TABLES / "Apply_predictions.csv")

    return {
        "df": df,
        "df_apply": df_apply,
        "target": target,
        "predictors": predictors,
        "categorical_predictors": categorical_predictors,
        "numeric_predictors": numeric_predictors,
        "categorical_options": categorical_options,
        "categorical_defaults": categorical_defaults,
        "numeric_defaults": numeric_defaults,
        "numeric_ranges": numeric_ranges,
        "df_predictors": df_predictors,
        "encoder": encoder,
        "scaling": scaling,
        "scaled_models": SCALED_MODELS,
        "models": MODELS,
        "model_mcc": model_mcc,
        "optimal_model": optimal_model,
        "optimal_model_name": optimal_model_name,
        "optimal_mcc": optimal_mcc,
        "test_predictions": test_predictions,
    }


if __name__ == "__main__":
    run_pipeline()
