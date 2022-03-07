import datetime
import logging
import os
import pickle
import time

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import roc_auc_score
from utils.modelagem import treinar_modelo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


@hydra.main(config_path="../config", config_name="config")
def train_model(cfg) -> None:
    # Constants
    # Stage
    STAGE = cfg.mlflow.stage
    # Description
    DESCRIPTION = cfg.mlflow.description
    # Register Model after training
    REGISTER_MODEL = cfg.mlflow.register_model
    # Step
    STEP = cfg.mlflow.step
    # Random State
    RANDOM_STATE = cfg.model.random_state
    # Model name
    MODEL_NAME = cfg.model.name
    # Tracking URI
    TRACKING_URI = cfg.mlflow.tracking_uri
    # Run name
    RUN_NAME = MODEL_NAME + str(datetime.datetime.now()).replace(" ", "T")
    # Input Data name
    INPUT_DATA = os.path.join(cfg.paths.inter, cfg.datasets.train_data)
    # Test Data nem
    TEST_DATA = os.path.join(cfg.paths.inter, cfg.datasets.test_data)
    # Artifact Path
    ARTIFACT_PATH = cfg.mlflow.artifact_path
    # Tag
    TAG = {"step": STEP, "register_model": REGISTER_MODEL}
    # Variaveis
    VARIAVEIS_CATEGORICAS = [
        "polyuria",
        "perda_de_peso_repentina",
        "genero",
        "polyphagia",
        "polydipsia",
        "paresia_parcial",
    ]
    VARIAVEIS_NUMERICAS = ["idade"]
    # Loading data and Logging
    try:
        df_train = pd.read_parquet(INPUT_DATA)
        df_test = pd.read_parquet(TEST_DATA)
        logger.info("Data for training and testing was read successfully")
    except Exception as error:
        logger.error("Data could not be read", exc_info=True)
        raise error

    # Training
    try:
        X = df_train[VARIAVEIS_CATEGORICAS + VARIAVEIS_NUMERICAS]
        y = df_train["target"]
        model, params = treinar_modelo(X, y)
        model.fit(X, y)
        y_predict = model.predict_proba(X)
        logger.info("Model trained sucessfully")
    except Exception as error:
        logger.error("Model could not be trained", exc_info=True)
        raise error
    try:
        # Testing
        X_test = df_test[VARIAVEIS_CATEGORICAS + VARIAVEIS_NUMERICAS]
        y_test = df_test["target"]
        y_predict_test = model.predict_proba(X_test)
        logger.info("Testing data was scored successfully")
    except Exception as error:
        logger.error("Testing data could not be scored")
    # Log model and params
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(MODEL_NAME)
        with mlflow.start_run(run_name=RUN_NAME) as run:
            mlflow.set_tags(TAG)
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "roc_auc_train": roc_auc_score(y, y_predict[:, 1]),
                    "roc_auc_test": roc_auc_score(y_test, y_predict_test[:, 1]),
                }
            )
            signature = infer_signature(X, model.predict_proba(X)[:, 1])
            mlflow.sklearn.log_model(
                model, artifact_path=ARTIFACT_PATH, signature=signature
            )
            logger.info("Model was logged in mlflow sucessfully")
            if REGISTER_MODEL == True:
                client = MlflowClient()
                RUN_ID = mlflow.active_run().info.run_id
                MODEL_URI = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
                MODEL_DETAILS = mlflow.register_model(
                    model_uri=MODEL_URI, name=MODEL_NAME
                )
                wait_until_ready(MODEL_DETAILS.name, MODEL_DETAILS.version)
                client.update_model_version(
                    name=MODEL_DETAILS.name,
                    version=MODEL_DETAILS.version,
                    description=DESCRIPTION,
                )
                client.transition_model_version_stage(
                    name=MODEL_DETAILS.name, version=MODEL_DETAILS.version, stage=STAGE
                )
                logger.info("Model was registered in mlflow sucessfully")

    except Exception as error:
        logger.error("Model could not be logged in mlflow", exc_info=True)
        raise error


if __name__ == "__main__":
    train_model()
