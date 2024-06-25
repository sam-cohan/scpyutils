"""
Convenience utilities for scikit-learn and preprocessing.

Author: Sam Cohan
"""

import re
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
from sklearn import calibration, metrics
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def force_dt_cols(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    """Given a DataFrame and list of date columns, enforce date type.

    Args:
        df: Input DataFrame.
        dt_cols: List of column names known to be date columns.

    Returns:
        Same as input dataframe (i.e. has side-effect).
    """
    for col in dt_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if not df[col].not_null().sum():
                print(f"WARNING: '{col}' is not a date column")
                continue
        print(f"####'{col}' has range {df[col].sort_values().values[[0, -1]]}")
    return df


def get_col_types(
    df: pd.DataFrame,
    known_num_cols: Optional[List[str]] = None,
    known_ctg_cols: Optional[List[str]] = None,
    known_dt_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Given a DataFrame return dictionary of types to column list.

    Args:
        df: Input DataFrame
        known_num_cols: List of known numeric columns. If provided column
            will not be checked.
        known_ctg_cols: List of known categorical columns. If provided column
            will not be checked.
        known_num_cols: List of known datetime columns. If provided column
            will not be checked.

    Returns:
        Dictionary mapping from a string which represents a type to a list
        of fields that have that type. Currently supports following types:
        {"num", "ctg", "dt"}.
    """
    known_num_cols = known_num_cols or set()
    known_ctg_cols = known_ctg_cols or set()
    known_dt_cols = known_dt_cols or set()
    df_cols = list(df)
    num_cols = [
        c
        for c in df_cols
        if c in known_num_cols or pd.api.types.is_numeric_dtype(df[c])
    ]
    ctg_cols = [
        c
        for c in df_cols
        if c in known_ctg_cols
        or (
            c not in num_cols
            and not re.search("date|created|timetamp", c, re.IGNORECASE)
        )
    ]
    dt_cols = [
        c
        for c in df_cols
        if c in known_dt_cols
        or (
            c not in num_cols
            and c not in ctg_cols
            and re.search("date|created|timetamp", c, re.IGNORECASE)
        )
    ]
    all_cols = set(num_cols) | set(ctg_cols) | set(dt_cols)
    assert sorted(all_cols) == sorted(
        df
    ), f"ERROR: unexpected result: {sorted(all_cols)} != {sorted(df)}"
    return dict(
        num=num_cols,
        ctg=ctg_cols,
        dt=dt_cols,
    )


def make_generic_preproc(
    df: pd.DataFrame, cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """Given a DataFrame and its columns, return a ColumnTransformer for
    preprocessing.

    The function will figure out field types and then set up a ColumnTransformer
    which would apply a median Imputer and StandardsScaler to numeric values and
    <NULL> Imputer and OneHotEncoder to categorical values.

    Args:
        df: input DataFrame
        cols: optional list of columns to target for preprocessing. If provided, any
           additional columns in the DataFrame are simply passed through at the end.
           This can be useful when you have a DataFrame which has both the features
           and the label data and you only want to transform the features. (default
           to None and will simply use all available columns)

     Returns:
        ColumnTransformer object which can be use to `fit_transform()` the input
        DataFrame. Note that the ColumnTransformer is not fitted.
    """
    if cols is None:
        cols = list(df)
    col_types = get_col_types(df[cols])
    if col_types["dt"]:
        print(f"WARNING: will ignore date columns: {col_types['dt']}")

    num_trans = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    ctg_trans = Pipeline(
        [
            # FIXME: It is probably not a good idea to assign a category to unknown.
            ("imputer", SimpleImputer(strategy="constant", fill_value="<NULL>")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # FIXME: Add pipeline for handling of dt_cols

    return ColumnTransformer(
        ([("num", num_trans, col_types["num"])] if col_types["num"] else [])
        + ([("ctg", ctg_trans, col_types["ctg"])] if col_types["ctg"] else []),
        remainder="passthrough",
    )


def fit_transform_generic_preproc(
    df: pd.DataFrame, cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, ColumnTransformer, List[str]]:
    """Given a DataFrame, apply generic preprocessing transform on it.

    This function makes use of `make_generic_preproc()` function, so check
    the details of the transformation there.

    Args:
        df: input DataFrame
        cols: optional list of columns to target for preprocessing. If provided, any
           additional columns in the DataFrame are simply passed through at the end.
           This can be useful when you have a DataFrame which has both the features
           and the label data and you only want to transform the features. (default
           to None and will simply use all available columns)

    Returns:
        Tuple containing the transformed data, the fitted ColumnTransformer, and a
        list of all the features which were transformed. Last argument is useful if
        you pass a subset of the columns as features).
    """
    preproc = make_generic_preproc(df, cols=cols)
    all_orig_cols = list(df)
    remain_cols = [c for c in all_orig_cols if c not in cols]
    df_trans = preproc.fit_transform(df)
    trans_cols = get_coltrans_cols(preproc)
    fitted_cols = trans_cols + remain_cols
    df_trans = pd.DataFrame(df_trans, columns=fitted_cols)
    return df_trans, preproc, trans_cols


def get_coltrans_cols(coltrans: ColumnTransformer) -> List[str]:
    """Given a ColumnTransformer, figures out the transformed output columns.

    Note that if the ColumnTransformer has `remainder="passthrough"` set,
    those will be missing from the list returned by this function.

    Args:
        coltrans (ColumnTransformer): sklearn
            ColumnTransformer that has been fit.

    Returns:
        (List[str]) List of transformed columns
    """
    columns = []
    # Note that we have to use the transformers_ member.
    # (the underscore is important as it is the fitted instance)
    for name, trans, cols in coltrans.transformers_:
        if isinstance(trans, Pipeline):
            ohes = [x for x in trans if isinstance(x, OneHotEncoder)]
            if not ohes:
                columns.extend(cols)
            else:
                assert len(ohes) == 1
                ohe = ohes[0]
                ohe_cols = ohe.get_feature_names()
                # It is possible that the transform above
                # removed the real column names...
                if not ohe_cols[0].startswith(cols[0]):
                    # Columns have been changed to x0_a, x0_b, etc.
                    col_idxs = [int(c.split("_", 1)[0][1:]) for c in ohe_cols]
                    ohe_cols = [
                        f"{cols[i]}_{c.split('_', 1)[1]}"
                        for c, i in zip(ohe_cols, col_idxs)
                    ]
                columns.extend(ohe_cols)
    return columns


def make_generic_model_pipeline(  # noqa: C901
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    model_class: BaseEstimator,
    params: Optional[
        Union[Dict[str, Union[str, float, List[Union[str, float]]]]]
    ] = None,  # noqa: E501
    cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
    scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
    refit: Optional[Union[bool, str, Callable]] = None,
    calib_method: Optional[str] = None,
    model_n_jobs: Optional[int] = -1,
    gs_n_jobs: Optional[int] = -1,
) -> Pipeline:
    """Make a generic model pipeline for structured data.

    This function is useful for making a generic sklearn model pipeline.
    Given a DataFrame of features and target variable, the desired model
    class and the params, it creates a generic preprocessing pipeline
    for the data that handles numeric and categorical variables, and
    depending on the whether the params are implying a grid-search,
    also includes a grid-search

    Args:
        df: DataFrame of features (x_col) and target variable (y_col)
        y_col: name of prediction target column.
        x_cols: list of feature column names.
        model_class: Any classifier or regressor sklearn object.
        params: either a dictionary of values, or a dictionary of list
            of values. If the latter, the model pipeline will be wrapped
            in a  GridSearch.
        cv: `cv` parameter from `sklearn.model_selection.GridSearch`
        scoring: `scoring` parameter from `sklearn.model_selection.GridSearch`
        refit: `refit` parameter from `sklearn.model_selection.GridSearch`
        calib_method: provide one of {"sigmoid", "isotonic"} if you are
            using a classifier which does not produce calibrated
            probabilities (i.e. non-GLM methods)
        model_n_jobs: `n_jobs` arg for the model (defaults to -1)
        gs_n_jobs: `n_jobs` arg for gridsearch (defaults to -1)

    Returns:
        Pipeline with names steps "preproc" and "model". Note that
        the pipeline is not fitted.
    """
    preproc = make_generic_preproc(df[x_cols + [y_col]], cols=x_cols)

    if params is None:
        params = {}

    params_is_grid = params and all([isinstance(val, list) for val in params.values()])

    if params_is_grid:
        model = model_class(n_jobs=model_n_jobs)
    else:
        model = model_class(**params, n_jobs=model_n_jobs)

    if calib_method:
        assert is_classifier(
            model
        ), f"ERROR: calibration not possible for non-classifier model {model}"
        print("will apply calibration to classifier output...")
        model = calibration.CalibratedClassifierCV(
            base_estimator=model, method=calib_method, cv=4
        )
        if params_is_grid:
            params = {f"base_estimator__{k}": v for k, v in params.items()}

    # If params is grid, then perform grid search using GridSearchCV
    if params_is_grid:
        if scoring is None:
            if is_regressor(model_class):
                scoring = {
                    "mse": metrics.make_scorer(metrics.mean_squared_error),
                }
                if refit is None:
                    refit = "mse"
            elif is_classifier(model_class):
                scoring = {
                    "roc_auc": metrics.make_scorer(
                        metrics.roc_auc_score, needs_proba=True
                    ),
                    "balanced_accuracy": metrics.make_scorer(
                        metrics.balanced_accuracy_score
                    ),
                }
                if refit is None:
                    refit = "roc_auc"
            else:
                raise Exception(
                    f"Model={model} is neither a regressor nor a classifier."
                )

        model = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring,
            refit=refit,
            cv=cv,
            return_train_score=True,
            n_jobs=gs_n_jobs,
        )

    return Pipeline([("preproc", preproc), ("model", model)])


def fit_predict_generic_model(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    model_class: BaseEstimator,
    params: Optional[
        Union[Dict[str, Union[str, float, List[Union[str, float]]]]]
    ] = None,  # noqa: E501
    cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
    scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
    refit: Optional[Union[bool, str, Callable]] = None,
    calib_method: Optional[str] = None,
    model_n_jobs: Optional[int] = -1,
    gs_n_jobs: Optional[int] = -1,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Create a generic model pipeline for structured data and fit and predict.

    For details of what the generic model pipeline does, please check
    `make_generic_model_pipeline`.

    Args:
        df: DataFrame of features (x_col) and target variable (y_col)
        y_col: name of prediction target column.
        x_cols: list of feature column names.
        model_class: Any classifier or regressor sklearn object.
        params: either a dictionary of values, or a dictionary of list
            of values. If the latter, the model pipeline will be wrapped
            in a  GridSearch.
        cv: `cv` parameter from `sklearn.model_selection.GridSearch`
        scoring: `scoring` parameter from `sklearn.model_selection.GridSearch`
        refit: `refit` parameter from `sklearn.model_selection.GridSearch`
        calib_method: provide one of {"sigmoid", "isotonic"} if you are
            using a classifier which does not produce calibrated
            probabilities (i.e. non-GLM methods)
        model_n_jobs: `n_jobs` arg for the model (defaults to -1)
        gs_n_jobs: `n_jobs` arg for gridsearch (defaults to -1)

    Returns:
        A tuple of the fitted model pipeline plus the predictions DataFrame.
    """
    model_pipeline = make_generic_model_pipeline(
        df=df,
        y_col=y_col,
        x_cols=x_cols,
        model_class=model_class,
        params=params,
        scoring=scoring,
        cv=cv,
        refit=refit,
        calib_method=calib_method,
        model_n_jobs=model_n_jobs,
        gs_n_jobs=gs_n_jobs,
    )
    print(f"fitting model for y_col={y_col} with params={params} ...")
    model_pipeline.fit(df[x_cols], df[y_col])
    model = model_pipeline.named_steps["model"]
    if isinstance(model, GridSearchCV):
        print(f"Best params:\n{model.best_params_}")

    preds = model_pipeline.predict(df[x_cols])
    preds_df = pd.DataFrame(preds, columns=[f"{y_col}_pred"])

    if is_regressor(model_class):
        print("MSE: ", metrics.mean_squared_error(df[y_col], preds_df))
    elif is_classifier(model_class):
        if hasattr(model_pipeline, "predict_proba"):
            preds_df[f"{y_col}_proba"] = model_pipeline.predict_proba(df[x_cols])[:, 1]
            print("AUC: ", metrics.roc_auc_score(df[y_col], preds_df[f"{y_col}_proba"]))
        else:
            print("predict_proba not available, will use decision_function...")
            preds_df[f"{y_col}_decifunc"] = model_pipeline.decision_function(df[x_cols])

        print("Confusion Matrix:\n", metrics.confusion_matrix(df[y_col], preds))
        print(metrics.classification_report(df[y_col], preds_df[f"{y_col}_pred"]))
    else:
        raise Exception(f"Model={model} is neither a regressor nor a classifier.")

    return model_pipeline, preds_df
