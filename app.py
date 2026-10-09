from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import run_pipeline
from anomaly_detection import run_anomaly_detection
from cross_dataset import UNSW_COLUMNS, run_cross_dataset
from robustness import run_robustness

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


@st.cache_data(show_spinner="Running the stress tests (first run only, cached after that)")
def get_robustness():
    return run_robustness(verbose=False)


@st.cache_data(show_spinner="Downloading UNSW-NB15 and running the cross-dataset tests (first run only)")
def get_cross_dataset():
    return run_cross_dataset(verbose=False)


@st.cache_data(show_spinner="Training the anomaly detectors (first run only, cached after that)")
def get_anomaly_detection():
    return run_anomaly_detection(verbose=False)


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


def render_stress_test():
    st.title("Is 99.7% too good to be true?")
    st.write(
        "A near-perfect score on an intrusion dataset usually means something is off: "
        "duplicated rows, information leaking from the test set, or a test that is "
        "simply too easy. This page checks each of those, then asks the question that "
        "matters in practice: how does the model handle an attack it has never seen?"
    )

    r = get_robustness()
    cv = r["cross_validation"].set_index(["Model", "Metric"])
    baselines = r["baselines"]
    curve = r["learning_curve"]
    unseen = r["unseen_attacks"]
    unseen_caught = (unseen["Recall when unseen"] * unseen["Connections"]).sum() / unseen["Connections"].sum()
    rows_for_094 = int(curve.loc[curve["MCC"] >= 0.94, "Training rows"].min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random Forest MCC, 15-fold CV", f"{cv.loc[('Random Forest', 'MCC'), 'Mean']:.3f}",
              f"± {cv.loc[('Random Forest', 'MCC'), 'Std']:.3f}", delta_color="off")
    c2.metric("Accuracy with a single split", f"{100 * baselines.loc[1, 'Accuracy']:.1f}%")
    c3.metric("Training rows for MCC 0.94", f"{rows_for_094:,}")
    c4.metric("Unseen attack families caught", f"{100 * unseen_caught:.1f}%")

    st.markdown("### 1. Is the score inflated by duplicates or leakage?")
    st.write(
        "No. The original KDD'99 data is famous for duplicated records that make test "
        "sets trivially easy; this NSL-KDD subset was cleaned of them. Not a single row "
        "of the held-out test split has an identical twin in the training split, and "
        "every encoder below is fit on the training part only."
    )
    st.dataframe(r["duplicates"], hide_index=True, width="stretch")
    st.write(
        "The score is also stable: repeated 5-fold cross-validation (3 repeats, 15 fits) "
        "gives the same number as the single 70/30 split, with a narrow spread."
    )
    st.dataframe(
        r["cross_validation"].style.format({c: "{:.4f}" for c in ["Mean", "Std", "CI low", "CI high"]}),
        hide_index=True, width="stretch",
    )

    st.markdown("### 2. Is the task simply easy?")
    st.write(
        "Yes, very. A decision tree with one split already gets "
        f"{100 * baselines.loc[1, 'Accuracy']:.1f}% accuracy, three levels reach MCC "
        f"{baselines.loc[3, 'MCC']:.2f}, and the full Random Forest only adds the last few points."
    )
    fig = go.Figure(go.Bar(
        x=baselines["MCC"], y=baselines["Model"], orientation="h",
        marker_color=["#94A3B8"] * (len(baselines) - 1) + ["#2563EB"],
        text=[f"{v:.3f}" for v in baselines["MCC"]], textposition="outside",
        hovertemplate="%{y}<br>MCC %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(t=10, b=40, l=10, r=40),
                      xaxis=dict(title="MCC on the test split", range=[0, 1.1]),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**The whole depth 2 tree** (MCC "
                    f"{baselines.loc[2, 'MCC']:.2f}, protocol_type 0 is ICMP)")
        st.code(r["depth2_rules"], language=None)
        st.caption(
            "In words: an almost empty payload to a rarely used service is an attack, "
            "and so is ICMP traffic carrying data. Four rules, "
            f"{100 * baselines.loc[2, 'Accuracy']:.1f}% accuracy."
        )
    with col_right:
        fig = go.Figure(go.Scatter(
            x=curve["Training rows"], y=curve["MCC"], mode="lines+markers",
            line=dict(color="#2563EB", width=2), marker=dict(size=8),
            hovertemplate="%{x:,} training rows<br>MCC %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(height=320, margin=dict(t=30, b=40, l=10, r=10),
                          title=dict(text="Learning curve, Random Forest", font=dict(size=14)),
                          xaxis=dict(title="Training rows (log scale)", type="log"),
                          yaxis=dict(title="MCC on the test split"))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"{rows_for_094} labeled connections, 1% of the training data, are already "
            "enough for MCC above 0.94."
        )

    single = r["single_predictor"].head(12)
    fig = go.Figure(go.Bar(
        x=single["MCC"][::-1], y=single["Predictor"][::-1], orientation="h",
        marker_color="#2563EB",
        hovertemplate="%{y}<br>MCC %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(t=30, b=40, l=10, r=10),
                      title=dict(text="What a small tree (depth 4) reaches using a single predictor",
                                 font=dict(size=14)),
                      xaxis=dict(title="MCC on the test split", range=[0, 1]))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 3. Does the model depend on a few predictors?")
    st.write(
        "No, the signal is spread out and redundant. Removing the most important "
        "predictors one group at a time barely moves the score until most of them are gone."
    )
    ablation = r["ablation"]
    fig = go.Figure(go.Scatter(
        x=ablation["Top predictors removed"], y=ablation["MCC"], mode="lines+markers",
        line=dict(color="#2563EB", width=2), marker=dict(size=8),
        customdata=ablation["Removed"],
        hovertemplate="Top %{x} removed<br>MCC %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(t=10, b=40, l=10, r=10),
                      xaxis=dict(title="Most important predictors removed"),
                      yaxis=dict(title="MCC on the test split"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 4. What about an attack the model has never seen?")
    st.write(
        "This is where the 99.7% falls apart. A random split puts examples of every "
        "attack type in both training and test, so the test only measures recognition "
        "of known attacks. The dataset has no attack type labels, so the anomalies were "
        "clustered into families (KMeans on log-scaled and one-hot encoded predictors). "
        "Each family was then removed from training entirely and the model was asked to "
        "detect it, which approximates a new kind of attack showing up in production."
    )
    labels = [f"Family {f}<br>{p}" for f, p in
              zip(unseen["Family"], unseen["Dominant pattern (protocol / service / flag)"])]
    fig = go.Figure()
    for column, name, color in [("Recall when seen in training", "Seen in training", "#2563EB"),
                                ("Recall when unseen", "Never seen", "#F97316")]:
        fig.add_bar(
            x=labels, y=100 * unseen[column], name=name, marker_color=color,
            text=[f"{100 * v:.0f}%" for v in unseen[column]], textposition="outside",
            hovertemplate=name + "<br>%{x}<br>%{y:.1f}% detected<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, height=440,
                      margin=dict(t=30, b=40, l=10, r=10),
                      yaxis=dict(title="Attacks detected (%)", range=[0, 112]),
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, width="stretch")
    st.write(
        f"Across all held-out families, only {100 * unseen_caught:.1f}% of the attacks "
        "are flagged, while the false positive rate on normal traffic stays around 0.1%. "
        "The model has not learned what an attack looks like in general; it has "
        "memorized the specific attacks in its training data and treats anything else "
        "as normal."
    )
    with st.expander("Full table"):
        st.dataframe(
            unseen.style.format({
                "Pattern share": "{:.0%}",
                "Recall when seen in training": "{:.1%}",
                "Recall when unseen": "{:.1%}",
                "False positive rate on normal traffic": "{:.2%}",
            }),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Pattern share is the fraction of the family that matches its dominant "
            "protocol, service and flag. Families smaller than 50 connections are skipped."
        )

    st.markdown("### Takeaways")
    st.write(
        "- The 99.7% is real for this test design: no duplicates, no leakage, stable across folds.\n"
        "- The test design is easy. A handful of rules gets most of the way, and very little data is needed.\n"
        "- Supervised models detect attacks they were trained on, not new ones. A real "
        "deployment would pair the classifier with anomaly detection trained on normal "
        "traffic only, and would evaluate on traffic from a different time or network."
    )


def render_cross_dataset():
    st.title("Does it work on another network?")
    st.write(
        "Every other page evaluates the model on the same dataset it was trained on. "
        "Here it meets UNSW-NB15, a dataset recorded in 2015 on a different network "
        "with modern attack tools, which is the situation a deployed intrusion detector "
        "is actually in."
    )

    r = get_cross_dataset()
    transfer = r["transfer"].set_index(["Trained on", "Tested on"])
    adversarial = r["adversarial"]
    adaptation = r["adaptation"]
    kdd_to_unsw = transfer.loc[("NSL-KDD", "UNSW-NB15")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MCC on NSL-KDD itself", f"{transfer.loc[('NSL-KDD', 'NSL-KDD'), 'MCC']:.3f}")
    c2.metric("MCC on UNSW-NB15", f"{kdd_to_unsw['MCC']:.3f}")
    c3.metric("UNSW-NB15 attacks detected", f"{100 * kdd_to_unsw['Attacks detected']:.2f}%")
    c4.metric("Networks told apart (AUC)", f"{adversarial.loc[0, 'AUC']:.2f}")

    st.markdown("### The two datasets")
    st.dataframe(r["datasets"].style.format({"Attack share": "{:.1%}", "Rows used": "{:,}",
                                             "Rows dropped": "{:,}"}),
                 hide_index=True, width="stretch")
    st.write(
        "The datasets use different feature sets, so the comparison uses the 8 "
        "attributes that measure the same thing in both. Service names and connection "
        "states were translated to the NSL-KDD vocabulary, and only tcp, udp and icmp "
        "connections are kept, since NSL-KDD has no other protocols."
    )
    with st.expander("Feature mapping"):
        st.dataframe(pd.DataFrame({"NSL-KDD": list(UNSW_COLUMNS.values()),
                                   "UNSW-NB15": list(UNSW_COLUMNS.keys())}),
                     hide_index=True, width="stretch")

    st.markdown("### 1. Train on one, test on the other")
    transfer_rows = r["transfer"]
    labels = [f"{a} → {b}" for a, b in zip(transfer_rows["Trained on"], transfer_rows["Tested on"])]
    fig = go.Figure(go.Bar(
        x=transfer_rows["MCC"], y=labels, orientation="h",
        marker_color=["#2563EB" if same else "#F97316" for same in transfer_rows["Same dataset"]],
        text=[f"{v:.2f}" for v in transfer_rows["MCC"]], textposition="outside",
        hovertemplate="%{y}<br>MCC %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(height=280, margin=dict(t=10, b=40, l=10, r=40),
                      xaxis=dict(title="MCC (0 = no better than chance)", range=[-0.4, 1.15],
                                 zeroline=True, zerolinecolor="#94A3B8"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue: trained and tested on the same dataset. Orange: tested on the other dataset.")
    st.write(
        "On its own data the model is near perfect even with only these 8 features. "
        f"On UNSW-NB15 it flags {100 * kdd_to_unsw['Attacks detected']:.2f}% of attacks, "
        "and its MCC is below zero: it does worse than guessing. The reverse direction "
        "fails as well."
    )

    st.markdown("### 2. Which attacks get through?")
    categories = r["categories"]
    attacks = categories[categories["Category"] != "Normal"]
    fig = go.Figure()
    for column, color in [("Trained on UNSW-NB15", "#2563EB"), ("Trained on NSL-KDD", "#F97316")]:
        fig.add_bar(
            x=attacks["Category"], y=100 * attacks[column], name=column, marker_color=color,
            text=[f"{100 * v:.0f}%" for v in attacks[column]], textposition="outside",
            customdata=attacks["Connections"],
            hovertemplate=column + "<br>%{x}: %{y:.1f}% detected<br>%{customdata:,} connections<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, height=420,
                      margin=dict(t=30, b=40, l=10, r=10),
                      yaxis=dict(title="UNSW-NB15 attacks detected (%)", range=[0, 112]),
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, width="stretch")
    st.write(
        "Every modern attack category is missed. The NSL-KDD attacks date from 1998 and "
        "are dominated by floods and scans; exploits, fuzzers and backdoors simply don't "
        "look like anything in its training data."
    )

    st.markdown("### 3. Why: the networks look nothing alike")
    st.write(
        "Adversarial validation: a classifier is trained to tell which dataset a "
        "**normal** connection came from. If the networks were similar, it would sit "
        "near 0.5 AUC. It separates them perfectly, and almost any single numeric feature "
        "is enough."
    )
    per_feature = adversarial.iloc[1:].sort_values("AUC")
    fig = go.Figure(go.Bar(
        x=per_feature["AUC"], y=per_feature["Features"], orientation="h", marker_color="#2563EB",
        text=[f"{v:.2f}" for v in per_feature["AUC"]], textposition="outside",
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94A3B8",
                  annotation_text="indistinguishable", annotation_position="bottom right")
    fig.update_layout(height=340, margin=dict(t=10, b=40, l=10, r=40),
                      xaxis=dict(title="AUC for telling the networks apart, one feature at a time",
                                 range=[0, 1.1]))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(r["distributions"].style.format(precision=2), hide_index=True, width="stretch")
    st.caption(
        "Normal traffic in NSL-KDD is tiny, instant connections to busy hosts; in "
        "UNSW-NB15 it carries kilobytes and the hosts see only a few connections. "
        "A rule like \"few bytes means attack\" is right on one network and wrong on the other."
    )

    st.markdown("### 4. Is the feature mapping to blame?")
    st.write(
        "The connection counters use different time windows in the two datasets, and "
        "UNSW-NB15 byte counts include packet headers. So the transfer was repeated "
        "without the counters, and with every numeric feature replaced by its percentile "
        "within its own dataset, which removes any difference in scale."
    )
    st.dataframe(
        r["sensitivity"].style.format({"MCC": "{:.3f}", "Accuracy": "{:.1%}",
                                       "Attacks detected": "{:.1%}", "False positive rate": "{:.1%}"}),
        hide_index=True, width="stretch",
    )
    st.write(
        "NSL-KDD → UNSW-NB15 fails under every variant, so the mapping is not the "
        "problem. One interesting asymmetry: after rank normalization, a model trained "
        "on UNSW-NB15 does transfer to NSL-KDD reasonably well. Training on a broad, "
        "modern set of attacks generalizes; training on a narrow, old one does not. "
        "(Rank normalization uses the unlabeled test traffic's own distribution, which "
        "a real deployment could also do.)"
    )

    st.markdown("### 5. How much local data fixes it?")
    fig = go.Figure()
    for setting, color, dash in [("NSL-KDD + local rows", "#F97316", "solid"),
                                 ("Local rows only", "#2563EB", "dot")]:
        part = adaptation[(adaptation["Training data"] == setting) & (adaptation["Labeled UNSW-NB15 rows"] > 0)]
        fig.add_scatter(
            x=part["Labeled UNSW-NB15 rows"], y=part["MCC"], name=setting, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
            hovertemplate=setting + "<br>%{x:,} labeled rows<br>MCC %{y:.3f}<extra></extra>",
        )
    fig.update_layout(height=380, margin=dict(t=30, b=40, l=10, r=10),
                      xaxis=dict(title="Labeled UNSW-NB15 connections in training (log scale)", type="log"),
                      yaxis=dict(title="MCC on the UNSW-NB15 testing set"),
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, width="stretch")
    st.write(
        "A few hundred labeled connections from the new network beat all 25,000 NSL-KDD "
        "rows, and adding NSL-KDD on top of local data changes MCC by at most about "
        "0.015 at any size. Once any local data is available, the public dataset adds "
        "essentially nothing."
    )

    st.markdown("### Takeaways")
    st.write(
        "- A 99.7% score on one dataset says almost nothing about another network.\n"
        "- The cause is distribution shift: both what normal traffic looks like and which "
        "attacks exist change between environments.\n"
        "- Labeled data from the target network matters far more than the amount of "
        "public training data. Any deployment needs local evaluation and periodic "
        "retraining, and benchmark scores should come from data the model's environment "
        "never saw."
    )


DETECTOR_COLORS = {
    "Isolation Forest": "#0D9488",
    "Local Outlier Factor": "#9333EA",
    "Autoencoder": "#2563EB",
    "Random Forest with local attack labels": "#F97316",
}


def tradeoff_chart(roc, colors, max_fpr):
    fig = go.Figure()
    for method, color in colors.items():
        part = roc[roc["Method"].str.startswith(method) & (roc["False positive rate"] <= max_fpr)]
        fig.add_scatter(
            x=100 * part["False positive rate"], y=100 * part["Attacks detected"], name=method,
            mode="lines", line=dict(color=color, width=2),
            hovertemplate=method + "<br>%{x:.1f}% false alarms<br>%{y:.1f}% attacks detected<extra></extra>",
        )
    fig.add_vline(x=100 * 0.01, line_dash="dash", line_color="#94A3B8",
                  annotation_text="1% target", annotation_position="top right")
    fig.update_layout(height=400, margin=dict(t=30, b=40, l=10, r=10), hovermode="closest",
                      xaxis=dict(title="Normal connections flagged (%)", range=[0, 100 * max_fpr]),
                      yaxis=dict(title="Attacks detected (%)", range=[0, 102]),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def render_anomaly_detection():
    st.title("Catching attacks it has never seen")
    st.write(
        "The two previous pages found the same weakness from two angles: a classifier "
        "trained on labeled attacks only recognizes those attacks. This page turns the "
        "question around. Anomaly detectors learn only what **normal** traffic looks like "
        "and flag anything that deviates, so they need no attack labels at all."
    )
    st.markdown(
        "- **Isolation Forest**: an unusual connection gets isolated by fewer random splits.\n"
        "- **Local Outlier Factor**: an unusual connection sits in a sparser region than its neighbors.\n"
        "- **Autoencoder**: a small neural network learns to compress and rebuild normal "
        "connections; the ones it rebuilds badly are unusual."
    )
    st.write(
        "Each detector's alarm threshold is set on held-out normal traffic so that about "
        "1% of normal connections raise an alarm. **Any detector** raises an alarm when "
        "at least one of the three does, with stricter individual thresholds so the "
        "total stays near 1%."
    )

    r = get_anomaly_detection()
    nsl = r["nsl_summary"].set_index("Method")
    unsw = r["unsw_summary"].set_index("Method")
    hybrid = "Random Forest + any detector"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unseen NSL-KDD attacks, Random Forest", f"{100 * nsl.loc['Random Forest', 'Attacks detected']:.1f}%")
    c2.metric("With anomaly detection added", f"{100 * nsl.loc[hybrid, 'Attacks detected']:.1f}%",
              f"{100 * nsl.loc[hybrid, 'False positive rate']:.1f}% false alarms", delta_color="off")
    c3.metric("UNSW-NB15, NSL-KDD model", f"{100 * unsw.loc['Random Forest trained on NSL-KDD', 'Attacks detected']:.2f}%")
    ae_local = unsw.loc["Autoencoder (local normal traffic)"]
    c4.metric("UNSW-NB15, autoencoder, no labels", f"{100 * ae_local['Attacks detected']:.1f}%",
              f"{100 * ae_local['False positive rate']:.1f}% false alarms", delta_color="off")

    st.markdown("### 1. Attack families held out of training (NSL-KDD)")
    st.write(
        "The same six attack families as on the \"Is 99.7% too good to be true?\" page. "
        "Each one was removed from the Random Forest's training data; the anomaly "
        "detectors never see any attacks at all."
    )
    families = r["nsl_families"]
    labels = [f"Family {f}<br>{p}" for f, p in
              zip(families["Family"], families["Dominant pattern (protocol / service / flag)"])]
    fig = go.Figure()
    for column, color in [("Random Forest", "#94A3B8"), ("Any detector", "#2563EB"), (hybrid, "#F97316")]:
        fig.add_bar(
            x=labels, y=100 * families[column], name=column, marker_color=color,
            text=[f"{100 * v:.0f}%" for v in families[column]], textposition="outside",
            customdata=families["Connections"],
            hovertemplate=column + "<br>%{x}<br>%{y:.1f}% detected<br>%{customdata:,} connections<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, height=440,
                      margin=dict(t=30, b=40, l=10, r=10),
                      yaxis=dict(title="Attacks detected (%)", range=[0, 112]),
                      legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        r["nsl_summary"].style.format({
            "Attacks detected": "{:.1%}", "Average over families": "{:.1%}",
            "False positive rate": "{:.2%}", "AUC": "{:.3f}",
        }, na_rep="-"),
        hide_index=True, width="stretch",
    )
    st.write(
        "Adding anomaly detection to the classifier raises detection of unseen attacks "
        f"from {100 * nsl.loc['Random Forest', 'Attacks detected']:.1f}% to "
        f"{100 * nsl.loc[hybrid, 'Attacks detected']:.1f}%, at the cost of "
        f"{100 * nsl.loc[hybrid, 'False positive rate']:.1f}% false alarms instead of "
        f"{100 * nsl.loc['Random Forest', 'False positive rate']:.1f}%. Most of the gain "
        "comes from the large SYN flood family (family 1), which looks nothing like normal "
        "traffic. The ICMP and HTTP families stay hidden: in these features they look like "
        "ordinary traffic, so no detector that only knows normal behavior can single them out."
    )
    with st.expander("Each detector separately, per family"):
        st.dataframe(
            families.style.format({c: "{:.1%}" for c in families.columns[3:]}),
            hide_index=True, width="stretch",
        )
        st.caption(
            "The detectors catch different families: Isolation Forest and the autoencoder "
            "find the floods, Local Outlier Factor finds part of the scans. Combining them "
            "with stricter thresholds did not beat the best single detector."
        )

    st.markdown("### 2. A new network with no labeled attacks (UNSW-NB15)")
    st.write(
        "Collecting normal traffic from a network is easy; collecting labeled attacks is "
        "not. So the detectors were trained only on UNSW-NB15's normal traffic, using all "
        "of its own features, and compared with the NSL-KDD Random Forest and with a "
        "Random Forest that has every local attack labeled (the best case, rarely available)."
    )
    summary = r["unsw_summary"]
    colors = ["#94A3B8" if "NSL-KDD" in m else "#F97316" if "attack labels" in m else "#2563EB"
              for m in summary["Method"]]
    fig = go.Figure(go.Bar(
        x=100 * summary["Attacks detected"], y=summary["Method"], orientation="h", marker_color=colors,
        text=[f"{100 * d:.1f}% detected, {100 * f:.1f}% false alarms"
              for d, f in zip(summary["Attacks detected"], summary["False positive rate"])],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f}% detected<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(t=10, b=40, l=10, r=10),
                      xaxis=dict(title="UNSW-NB15 attacks detected (%)", range=[0, 160]),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Gray: trained on NSL-KDD. Blue: trained on local normal traffic only. "
        "Orange: trained with labeled local attacks."
    )
    st.write(
        "The autoencoder, given nothing but normal traffic from the new network, detects "
        f"{100 * ae_local['Attacks detected']:.1f}% of attacks where the NSL-KDD model "
        "detects almost none. Normal-only training still has to be local, though: the same "
        "autoencoder trained on NSL-KDD's normal traffic flags "
        f"{100 * unsw.loc['Autoencoder on NSL-KDD normal traffic', 'False positive rate']:.0f}% "
        "of UNSW-NB15's normal traffic, which makes it useless."
    )

    st.markdown("**The trade-off between missed attacks and false alarms**")
    st.write(
        "A single threshold can be misleading: the labeled Random Forest above detects "
        "more but also raises far more false alarms. Moving each method's threshold "
        "gives the full picture."
    )
    st.plotly_chart(tradeoff_chart(r["unsw_roc"], DETECTOR_COLORS, max_fpr=0.2), width="stretch")
    st.write(
        "With labeled attacks, the Random Forest is still the best at any false alarm "
        "rate. Among the label-free detectors the autoencoder clearly leads (AUC "
        f"{ae_local['AUC']:.2f} versus {unsw.loc['Random Forest with local attack labels', 'AUC']:.2f} "
        "with labels). Labels help, but the autoencoder gets most of the way without any."
    )
    categories = r["unsw_categories"]
    with st.expander("Detection per UNSW-NB15 attack category"):
        st.dataframe(
            categories.style.format({c: "{:.0%}" for c in categories.columns[2:]})
            .background_gradient(cmap="Blues", subset=list(categories.columns[2:]), vmin=0, vmax=1),
            hide_index=True, width="stretch",
        )

    st.markdown("### Takeaways")
    st.write(
        "- Anomaly detection catches a large share of attacks that a classifier has never "
        "seen, and it only needs normal traffic, which every network has plenty of.\n"
        "- It is not a replacement: attacks that resemble normal traffic slip through, and "
        "a classifier trained on labeled attacks is more precise for the attacks it knows.\n"
        "- The practical design is both together: the classifier for known attacks, the "
        "anomaly detector as a safety net for new ones, trained on the target network's "
        "own traffic.\n"
        "- No detector wins everywhere (Isolation Forest is best on NSL-KDD, the "
        "autoencoder on UNSW-NB15), and picking one requires at least a few labeled "
        "attacks to validate on. Choosing the winner from these test results would repeat "
        "the mistake this project set out to expose."
    )


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
    st.write(
        "The bigger caveat is the test design. A random split only measures how well "
        "the model recognizes attack types it has already seen. When a whole attack "
        "family is held out of training, Random Forest detects only about 10% of it "
        "(see \"Is 99.7% too good to be true?\"). On a different network, UNSW-NB15, "
        "it detects almost none of the attacks (see \"Does it work on another network?\")."
    )
    st.write(
        "Anomaly detectors trained only on normal traffic close much of that gap: combined "
        "with the Random Forest they catch about 70% of the unseen attack families, and on "
        "UNSW-NB15 an autoencoder trained on local normal traffic detects about 66% of "
        "attacks without a single attack label (see \"Catching attacks it has never seen\")."
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
        ["Overview", "Predictor analysis", "Model comparison", "Model details",
         "Is 99.7% too good to be true?", "Does it work on another network?",
         "Catching attacks it has never seen", "Live prediction", "Conclusions"],
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
    elif page == "Is 99.7% too good to be true?":
        render_stress_test()
    elif page == "Does it work on another network?":
        render_cross_dataset()
    elif page == "Catching attacks it has never seen":
        render_anomaly_detection()
    elif page == "Live prediction":
        render_prediction(results)
    elif page == "Conclusions":
        render_conclusions(results)


if __name__ == "__main__":
    main()
