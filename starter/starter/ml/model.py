import json

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> sklearn.base.ClassifierMixin:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_sliced_performance(X, y, groups, model: sklearn.base.ClassifierMixin):
    preds = inference(model, X)
    sliced_metrics = {}

    for group_val in set(groups):
        mask = groups == group_val
        precision, recall, fbeta = compute_model_metrics(y[mask], preds[mask])
        sliced_metrics[group_val] = {'precision': precision, 'recall': recall, 'fbeta': fbeta, 'size': sum(mask)}
    return sliced_metrics


def inference(model: sklearn.base.ClassifierMixin, X: np.ndarray):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.base.ClassifierMixin
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model: sklearn.base.ClassifierMixin, savepath: str):
    joblib.dump(model, savepath)


def save_encoder(encoder: sklearn.base.TransformerMixin, savepath: str):
    joblib.dump(encoder, savepath)


def save_metrics(metrics: dict, filepath: str):
    with open(filepath, mode='w') as f:
        json.dump(metrics, f, indent=4)
