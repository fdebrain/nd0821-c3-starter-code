import logging
import tempfile
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from starter.ml.model import inference, save_model, train_model

LOG_SAVEDIR = Path("logs")
LOG_SAVEDIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_SAVEDIR / 'test_model.log',
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    force=True)


def test_train_model(processed_data):
    X, y = processed_data

    try:
        _ = train_model(X, y)
        logging.info("Testing train_model: Success")
    except Exception as e:
        logging.error("Testing train_model: Failed")
        raise e


def test_inference(processed_data, model):
    X, _ = processed_data

    try:
        preds = inference(model, X)
        logging.info("Testing inference: Success")
    except Exception as e:
        logging.error("Testing inference: Failed")
        raise e

    try:
        assert len(preds) == len(X)
    except AssertionError:
        logging.error("Testing inference: Unexpected output")


def test_save_model(model):
    with tempfile.TemporaryDirectory() as temp_dir:
        savepath = Path(temp_dir) / 'model.pkl'
        try:
            save_model(model, savepath)
            logging.info("Testing save_model: Success")
        except Exception as e:
            logging.error("Testing save_model: Failed")
            raise e

        try:
            assert savepath.exists(), "Model was not saved correctly"
        except AssertionError:
            logging.error("Testing save_model: Unexpected output")
