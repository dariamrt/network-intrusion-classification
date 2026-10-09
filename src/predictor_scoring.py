import numpy as np
import pandas as pd
from mrmr.pandas import mrmr_classif
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, StandardScaler
from sklearn.utils.multiclass import type_of_target

from src.config import TABLES


def mi_relevance(X, y):
    mi = mutual_info_classif(X, y, random_state=0)
    return pd.Series(mi, index=X.columns)


def mi_redundancy(X, target_column, features):
    scores = {}
    for f in features:
        mi = mutual_info_classif(X[[f]], X[target_column])[0]
        scores[f] = mi
    return pd.Series(scores)


def iv_woe(t, predictors, target):
    iv = pd.DataFrame(data={"Variable": predictors})
    classes = np.unique(t[target])
    q = len(classes)
    n = len(t)
    if q == 2:
        # binarize target for compatibility with calculate_iv_woe
        y_bin = (t[target] == classes[1]).astype(float).values
        iv_ = calculate_iv_woe(t, predictors, y_bin)
        iv = iv.merge(iv_)
    else:
        for cls in classes:
            y = np.zeros(n)
            y[t[target] == cls] = 1
            iv_ = calculate_iv_woe(t, predictors, y, cls)
            iv = iv.merge(iv_)
        iv["IV"] = iv.iloc[:, 1:].mean(axis=1)
    return iv


def calculate_iv_woe(t, predictors, y, class_=None, bins=10, output_woe=None):
    woe = pd.DataFrame()
    iv = pd.DataFrame()
    for v in predictors:
        if is_numeric_dtype(t[v]) and (len(np.unique(t[v])) > bins):
            binned_x = pd.qcut(t[v], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': y})
        else:
            d0 = pd.DataFrame({'x': t[v], 'y': y})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']) - np.log(d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=v)
        iv_field = "IV" if class_ is None else "IV_" + str(class_)
        temp = pd.DataFrame({"Variable": [v], iv_field: [d['IV'].sum()]})
        iv = pd.concat([iv, temp], axis=0)
        woe = pd.concat([woe, d], axis=0)
    if output_woe is not None:
        woe.to_csv(output_woe)
    return iv


def filtering(df: pd.DataFrame, predictors: np.ndarray,
              categorical_predictors: np.ndarray, numeric_predictors: np.ndarray, target):
    y = df[target].values

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if len(categorical_predictors) > 0:
        x_ = encoder.fit_transform(df[categorical_predictors])
        for i in range(len(categorical_predictors)):
            df[categorical_predictors[i]] = x_[:, i]

    x = df[predictors].values

    df_scores = pd.DataFrame(index=predictors)
    df_ranks = pd.DataFrame()
    df_tests = pd.DataFrame(index=predictors)

    # Mutual information
    mi = mutual_info_classif(x, y, random_state=0)
    df_scores["MI"] = mi
    k = np.flip(np.argsort(mi))
    df_ranks["MI"] = predictors[k]

    # Fisher
    f_scores, f_prob = f_classif(x, y)
    df_scores["F"] = f_scores
    k = np.flip(np.argsort(f_scores))
    df_ranks["F"] = predictors[k]
    df_tests["F"] = f_prob

    # Chi2
    df_ = df.copy()
    encoding = KBinsDiscretizer(n_bins=int(np.log2(len(df))) + 1,
                                 encode="ordinal", random_state=0)
    x_ = encoding.fit_transform(df_[numeric_predictors])
    for i in range(len(numeric_predictors)):
        df_[numeric_predictors[i]] = x_[:, i]
    ch2_stat, chi2_prob = chi2(df_[predictors], y)
    df_scores["Chi2"] = ch2_stat
    df_tests["Chi2"] = chi2_prob
    k = np.flip(np.argsort(ch2_stat))
    df_ranks["Chi2"] = predictors[k]

    # mRMR -> exclude columns with 0 variance (mrmr removes them automatically)
    nonzero_predictors = np.array([p for p in predictors if df[p].var() > 0])
    has_continuous_predictor = any(type_of_target(df[p]) == "continuous" for p in nonzero_predictors)
    if has_continuous_predictor:
        mrmr_ranks, mrmr_relevance, mrmr_redundancy = mrmr_classif(
            df[nonzero_predictors], df[target], K=len(nonzero_predictors),
            show_progress=False, return_scores=True
        )
    else:
        mrmr_ranks, mrmr_relevance, mrmr_redundancy = mrmr_classif(
            df[nonzero_predictors], df[target], K=len(nonzero_predictors),
            redundancy=mi_redundancy, relevance=mi_relevance,
            show_progress=False, return_scores=True
        )
    # align to the full list of predictors (0-variance columns get NaN)
    mrmr_score_dict = dict(zip(mrmr_ranks,
                                mrmr_relevance / mrmr_redundancy.sum(axis=1)))
    df_scores["mrmr"] = [mrmr_score_dict.get(p, np.nan) for p in predictors]
    # mrmr ranks, 0-variance columns are appended at the end
    zero_predictors = [p for p in predictors if p not in nonzero_predictors]
    df_ranks["mrmr"] = list(mrmr_ranks) + zero_predictors

    # information value
    iv = iv_woe(df, predictors, target)
    df_scores["IV"] = iv["IV"].values
    df_ranks["IV"] = predictors[np.flip(iv["IV"].argsort())]

    df_scores.to_csv(TABLES / "filtering_scores.csv")
    df_ranks.to_csv(TABLES / "filtering_ranks.csv")
    df_tests.to_csv(TABLES / "filtering_tests.csv")

    # standardized mean score
    scaler = StandardScaler()
    std_scores = scaler.fit_transform(df_scores)
    mean_scores = np.nanmean(std_scores, axis=1)
    k = np.flip(np.argsort(mean_scores))
    df_mean_scores = pd.DataFrame()
    df_mean_scores["Predictors"] = predictors[k]
    df_mean_scores["Standardized mean score"] = mean_scores[k]

    return df_mean_scores, x, encoder
