from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import run_pipeline

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

TABLES = Path("data_out/tables")
PLOTS = Path("data_out/plots")

MODEL_LABELS = {
    "GaussianNB": "Gaussian Naive Bayes",
    "KDE": "Kernel Density Naive Bayes",
    "LDA": "Linear Discriminant Analysis",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "Bagging": "Bagging",
    "ADABoost": "AdaBoost",
    "SVMLin": "Linear SVM",
    "SVM_Gaussian": "Gaussian SVM",
    "kNN": "k-Nearest Neighbors",
    "RegL": "Logistic Regression",
}

# quick notes on each model, shown on the "Model details" page
MODEL_BLURBS = {
    "GaussianNB": "Assumes predictors are normally distributed and independent given the class.",
    "KDE": "Same idea as Naive Bayes, but estimates each predictor's density with a Gaussian "
           "kernel instead of assuming normality.",
    "LDA": "Linear decision boundary that maximizes between-class vs. within-class variance.",
    "DT": "Splits on Gini impurity. Fully interpretable rules, easy to overfit.",
    "RF": "Bagged decision trees, each split considers a random subset of predictors.",
    "Bagging": "Bagged decision trees, no random feature subset (that's what makes RF different).",
    "ADABoost": "Sequential ensemble, reweights misclassified points so the next model focuses on them.",
    "SVMLin": "Max-margin linear separator. Probabilities come from Platt scaling on top.",
    "SVM_Gaussian": "RBF kernel maps the data into a higher dimension where it's linearly separable.",
    "kNN": "Majority vote among the k nearest neighbors, features standardized first.",
    "RegL": "Sigmoid applied to a linear combination of the predictors.",
}


@st.cache_resource(show_spinner="Training all 11 models (first run only, cached after that)")
def get_results():
    return run_pipeline(verbose=False)


def read_accuracy(model_name):
    return pd.read_csv(TABLES / f"accuracy_{model_name}.csv", index_col=0)


def read_errors(model_name):
    return pd.read_csv(TABLES / f"errors_{model_name}.csv", index_col=0)


@st.cache_data
def build_comparison_table(models):
    rows = []
    for m in models:
        acc = read_accuracy(m)
        n_errors = len(read_errors(m))
        rows.append({
            "Model": MODEL_LABELS[m],
            "Code": m,
            "Global accuracy (%)": round(float(acc.loc["Global accuracy"].iloc[0]), 2),
            "Mean accuracy (%)": round(float(acc.loc["Mean accuracy"].iloc[0]), 2),
            "CK index": round(float(acc.loc["CK index"].iloc[0]), 3),
            "MCC": round(float(acc.loc["MCC"].iloc[0]), 3),
            "Errors": n_errors,
        })
    table = pd.DataFrame(rows).sort_values("MCC", ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    return table


def render_overview(results):
    df = results["df"]
    df_apply = results["df_apply"]
    target = results["target"]

    st.title("Network Traffic Classification for Intrusion Detection")
    st.caption("Comparing 11 classifiers on the task of telling normal traffic apart from attacks.")

    class_counts = df[target].value_counts()
    n_normal = int(class_counts.get("normal", 0))
    n_anomaly = int(class_counts.get("anomaly", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training instances", f"{len(df):,}")
    c2.metric("Apply instances", f"{len(df_apply):,}")
    c3.metric("Predictors", len(results["predictors"]))
    c4.metric(
        "Optimal model",
        MODEL_LABELS[results["optimal_model_name"]],
        f"MCC {results['optimal_mcc']:.3f}",
    )

    st.markdown("### Problem statement")
    st.write(
        "Given the attributes of a network connection, predict whether it's normal or "
        "an intrusion/anomaly. Framed as binary classification."
    )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("### Dataset")
        st.write(
            "Kaggle's Network Intrusion Detection dataset, 41 predictors per connection: "
            "protocol type, service, error rates, host activity counters and so on. "
            "3 predictors are categorical (protocol_type, service, flag), the rest numeric. "
            "num_outbound_cmds and is_host_login have zero variance in the training data - "
            "kept them anyway since a constant column doesn't hurt the models."
        )
        st.write(
            f"{len(df):,} labeled connections for training, {len(df_apply):,} unlabeled "
            "ones in the apply set that the optimal model classifies at the end."
        )
    with col_right:
        st.markdown("### Class balance")
        fig = px.pie(
            names=["Normal", "Anomaly"],
            values=[n_normal, n_anomaly],
            color=["Normal", "Anomaly"],
            color_discrete_map={"Normal": "#2563EB", "Anomaly": "#F97316"},
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Methodology")
    st.write(
        "Predictors get scored by five filters before training: mutual information, "
        "Fisher's F test, Chi-squared, mRMR and information value (IV/WOE). Categorical "
        "predictors are ordinal-encoded, with unseen categories mapped to their own code "
        "so the same encoder still works on the apply set."
    )
    st.write(
        "70/30 stratified train/test split. All 11 classifiers are trained on the same "
        "split and evaluated on the same held-out test set; the one with the highest "
        "Matthews correlation coefficient (MCC) gets picked as optimal, since MCC uses "
        "all four confusion matrix cells and doesn't get skewed by class imbalance."
    )


def render_predictors(results):
    st.title("Predictor analysis")
    st.write(
        "Each predictor gets scored against the target with the five criteria above, "
        "standardized and averaged into one ranking. Used here as a reference and to "
        "read the tree models' feature importance later - nothing was dropped from "
        "training based on this."
    )

    df_predictors = results["df_predictors"]
    top_n = st.slider("Predictors to display", min_value=5, max_value=len(df_predictors), value=15)
    top = df_predictors.head(top_n)

    fig = go.Figure(
        go.Bar(
            x=top["Standardized mean score"][::-1],
            y=top["Predictors"][::-1],
            orientation="h",
            marker_color="#2563EB",
        )
    )
    fig.update_layout(
        height=max(320, 26 * top_n),
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis_title="Standardized mean score",
        yaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Underlying scores (mutual information, Fisher F, Chi squared, mRMR, IV)"):
        scores = pd.read_csv(TABLES / "filtering_scores.csv", index_col=0)
        st.dataframe(scores, width="stretch")

    st.caption(
        "src_bytes ranks near the top across every criterion here, and turns out to be "
        "the single most important feature in the tree models too."
    )


def render_comparison(results):
    st.title("Model comparison")
    st.write(
        "All 11 models, same 70/30 split, ranked by MCC below since that's the "
        "criterion used to pick the optimal one."
    )

    table = build_comparison_table(results["models"])
    optimal_code = results["optimal_model_name"]

    def highlight_optimal(row):
        color = "background-color: rgba(37, 99, 235, 0.12)" if row["Code"] == optimal_code else ""
        return [color] * len(row)

    st.dataframe(
        table.style.apply(highlight_optimal, axis=1).format(
            {"Global accuracy (%)": "{:.2f}", "Mean accuracy (%)": "{:.2f}", "CK index": "{:.3f}", "MCC": "{:.3f}"}
        ),
        width="stretch",
    )

    fig = go.Figure(
        go.Bar(
            x=table["Model"],
            y=table["MCC"],
            marker_color=["#2563EB" if c == optimal_code else "#94A3B8" for c in table["Code"]],
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(t=20, b=80, l=10, r=10),
        yaxis_title="MCC",
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"{MODEL_LABELS[optimal_code]} wins: MCC {results['optimal_mcc']:.3f}, "
        "fewest classification errors on the test set."
    )


def render_model_details(results):
    st.title("Model details")

    models = results["models"]
    labels = [MODEL_LABELS[m] for m in models]
    default_index = models.index(results["optimal_model_name"])
    choice = st.selectbox("Select a model", labels, index=default_index)
    model_code = models[labels.index(choice)]

    st.write(MODEL_BLURBS[model_code])

    acc = read_accuracy(model_code)
    n_errors = len(read_errors(model_code))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global accuracy", f"{float(acc.loc['Global accuracy'].iloc[0]):.2f}%")
    c2.metric("Mean accuracy", f"{float(acc.loc['Mean accuracy'].iloc[0]):.2f}%")
    c3.metric("CK index", f"{float(acc.loc['CK index'].iloc[0]):.3f}")
    c4.metric("MCC", f"{float(acc.loc['MCC'].iloc[0]):.3f}")
    st.caption(f"{n_errors} misclassified on the test set.")

    tab_names = ["Confusion matrix", "ROC curve", "Gain and lift", "2D scatter"]
    if model_code in ("DT", "RF"):
        tab_names.insert(0, "Feature importance")
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    if "Feature importance" in tab_map:
        with tab_map["Feature importance"]:
            st.image(str(PLOTS / f"FI_{model_code}_{model_code}.png"), width="stretch")
            if model_code == "DT":
                st.image(str(PLOTS / "DTree.png"), caption="Decision tree, first three levels", width="stretch")

    with tab_map["Confusion matrix"]:
        st.image(str(PLOTS / f"CM_{model_code}.png"), width="stretch")

    with tab_map["ROC curve"]:
        st.image(str(PLOTS / f"ROC_{model_code}.png"), width="stretch")

    with tab_map["Gain and lift"]:
        gain_files = sorted(PLOTS.glob(f"Gain_*_{model_code}.png"))
        lift_files = sorted(PLOTS.glob(f"Lift_*_{model_code}.png"))
        col1, col2 = st.columns(2)
        for path in gain_files:
            col1.image(str(path), width="stretch")
        for path in lift_files:
            col2.image(str(path), width="stretch")

    with tab_map["2D scatter"]:
        st.caption("Test set instances reduced to two dimensions with PCA.")
        scatter_cols = st.columns(3)
        for col, suffix, caption in zip(
            scatter_cols,
            ["classes", "predictions", "errors"],
            ["Actual classes", "Predicted classes", "Correct versus incorrect"],
        ):
            col.image(str(PLOTS / f"Plot_{model_code}_{suffix}.png"), caption=caption, width="stretch")


def render_prediction(results):
    st.title("Live prediction")
    st.write(
        "Enter values for the predictors that matter most to the optimal model. "
        "Everything else gets filled in from its typical training value, so the model "
        "still sees a complete feature vector."
    )

    optimal_model = results["optimal_model"]
    optimal_model_name = results["optimal_model_name"]
    predictors = results["predictors"]
    categorical_predictors = list(results["categorical_predictors"])
    numeric_defaults = results["numeric_defaults"]
    numeric_ranges = results["numeric_ranges"]
    categorical_options = results["categorical_options"]
    categorical_defaults = results["categorical_defaults"]
    encoder = results["encoder"]
    scaling = results["scaling"]
    scaled_models = results["scaled_models"]

    fi = pd.read_csv(TABLES / "FI_RF.csv").sort_values("MDI", ascending=False)
    top_features = fi["Predictors"].head(10).tolist()

    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(top_features):
        col = cols[i % 2]
        if feature in categorical_predictors:
            options = categorical_options[feature]
            default = categorical_defaults[feature]
            values[feature] = col.selectbox(
                feature, options, index=options.index(default) if default in options else 0
            )
        else:
            lo, hi = numeric_ranges[feature]
            default = float(numeric_defaults[feature])
            values[feature] = col.number_input(
                feature, min_value=lo, max_value=hi, value=min(max(default, lo), hi)
            )

    if st.button("Classify connection", type="primary"):
        row = {}
        for p in predictors:
            if p in values:
                row[p] = values[p]
            elif p in categorical_predictors:
                row[p] = categorical_defaults[p]
            else:
                row[p] = numeric_defaults[p]

        input_df = pd.DataFrame([row])
        if len(categorical_predictors) > 0:
            encoded = encoder.transform(input_df[categorical_predictors])
            for i, c in enumerate(categorical_predictors):
                input_df[c] = encoded[:, i]

        x_input = input_df[predictors].values
        if optimal_model_name in scaled_models:
            x_input = scaling.transform(x_input)

        prediction = optimal_model.predict(x_input)[0]
        if hasattr(optimal_model, "predict_proba"):
            proba = optimal_model.predict_proba(x_input)[0]
            classes = list(optimal_model.classes_)
            confidence = proba[classes.index(prediction)]
        else:
            confidence = None

        if prediction == "normal":
            st.success("Predicted class: normal")
        else:
            st.error("Predicted class: anomaly")
        if confidence is not None:
            st.caption(f"Model confidence: {confidence * 100:.1f}% ({MODEL_LABELS[optimal_model_name]})")


def render_conclusions(results):
    st.title("Conclusions")
    st.write(
        "Random Forest comes out on top: 99.68% global accuracy, MCC 0.994 on the "
        "held-out test set. Makes sense given it can pick up non-linear interactions "
        "between predictors that the linear and Bayesian models can't."
    )
    st.write(
        "src_bytes, dst_bytes and same_srv_rate are the top features by importance - "
        "data volume and connection consistency turn out to be the strongest signals "
        "for telling anomalies apart from normal traffic."
    )
    st.write(
        "LDA, linear SVM and logistic regression land at MCC 0.80-0.91: decent, but "
        "behind the tree models, which suggests the real decision boundary isn't linear "
        "in the original feature space. Gaussian Naive Bayes does worst since its "
        "independence/normality assumptions don't hold for this data; KDE fixes that by "
        "dropping the normality assumption and improves on it by a wide margin."
    )
    st.write(
        "One caveat: KDE and Gaussian SVM were trained and evaluated on stratified "
        "subsamples, not the full dataset, purely for runtime reasons. Worth rerunning "
        "on the full set with more compute at some point."
    )
    st.markdown("### Reference")
    st.write(
        "Sampada Bhosale, Network Intrusion Detection Dataset. "
        "[kaggle.com/datasets/sampadab17/network-intrusion-detection]"
        "(https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)"
    )


def main():
    st.sidebar.title("Network Intrusion Detection")
    st.sidebar.caption("Classification study and interactive demo")
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Predictor analysis", "Model comparison", "Model details", "Live prediction", "Conclusions"],
        label_visibility="collapsed",
    )

    results = get_results()

    if page == "Overview":
        render_overview(results)
    elif page == "Predictor analysis":
        render_predictors(results)
    elif page == "Model comparison":
        render_comparison(results)
    elif page == "Model details":
        render_model_details(results)
    elif page == "Live prediction":
        render_prediction(results)
    elif page == "Conclusions":
        render_conclusions(results)


if __name__ == "__main__":
    main()
