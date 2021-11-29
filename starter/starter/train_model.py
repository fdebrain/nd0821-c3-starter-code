# Script to train machine learning model.
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (compute_model_metrics,
                              compute_sliced_performance, inference,
                              save_encoder, save_metrics, save_model,
                              train_model)

LOG_SAVEDIR = Path("logs")
LOG_SAVEDIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_SAVEDIR / 'train_model.log',
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    force=True)


def main():
    logging.info('Load and split data')
    data = pd.read_csv('data/census_cleaned.csv')
    train, test = train_test_split(data, test_size=0.20)

    logging.info('Process training data')
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(train,
                                                 categorical_features=cat_features,
                                                 label="salary",
                                                 training=True)

    logging.info('Training model')
    model = train_model(X_train, y_train)

    logging.info('Process and infer on test set')
    X_test, y_test, _, _ = process_data(test,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)
    y_preds = inference(model, X_test)

    logging.info('Compute metrics from test set')
    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
    logging.info(
        f"Precision: {precision:.2f} | Recall: {recall:.2f} | F-beta: {fbeta:.2f}")

    logging.info('Compute sliced metrics for all categorical columns')
    for col in cat_features:
        groups = test[col]
        sliced_metrics = compute_sliced_performance(X_test, y_test, groups, model)
        save_metrics(sliced_metrics, f'model/metrics_by_{col}.json')

    logging.info('Save model and encoder')
    save_encoder(encoder, 'model/encoder.pkl')
    save_model(model, 'model/trained_model.pkl')


if __name__ == '__main__':
    main()
