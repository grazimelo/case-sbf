import json
import logging
import os
import pickle
import warnings

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking.client import MlflowClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="./config", config_name="config")
def config_app(cfg) -> None:
    global HOST
    global PORT
    global DEBUG
    global latest_model
    # Constants
    # Host
    HOST = cfg.app.host
    # Port
    PORT = cfg.app.port
    # Debug Mode
    DEBUG = cfg.app.host
    # Model name
    MODEL_NAME = cfg.model.name
    # Tracking uri
    TRACKING_URI = cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        latest_version_model = client.get_latest_versions(
            name=MODEL_NAME, stages=["Production"]
        )
        model_version_uri = f"models:/{MODEL_NAME}/{latest_version_model[0].version}"

        latest_model = mlflow.pyfunc.load_model(model_version_uri)
        logger.info("Latest model version read from mlflow production stage")
    except Exception as error:
        logger.error("Could not load the latest model version", exc_info=True)


# initialize flask application
app = Flask(__name__)


@app.route("/")
def home():
    return "App is Running"


@app.route("/predict", methods=["GET"])
def predict():
    X = request.get_json(force=True)
    X = pd.DataFrame(X, index=[0])
    prediction = str(latest_model.predict(X)[0])
    results = {"prediction": prediction}
    return json.dumps(results)


if __name__ == "__main__":
    config_app()
    # run web server
    app.run(host=HOST, debug=DEBUG, port=PORT)
