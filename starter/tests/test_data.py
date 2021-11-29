
import logging
from pathlib import Path

from starter.ml.data import process_data

LOG_SAVEDIR = Path("logs")
LOG_SAVEDIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_SAVEDIR / 'test_data.log',
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    force=True)


def test_process_training(data, cat_features):

    try:
        X, y, _, _ = process_data(data,
                                  cat_features,
                                  label="salary",
                                  training=True)
        logging.info("Testing process_data: Success")
    except Exception as e:
        logging.error("Testing process_data: Failed")
        raise e

    try:
        assert len(data) == len(X), \
            'Number of observations should remain the same after processing.'
        assert y.ndim == 1, \
            'y should be a vector.'
        assert len(X) == len(y), \
            'X and y should have the same number of observations.'
    except AssertionError:
        logging.error("Testing process_data: Unexpected output")
