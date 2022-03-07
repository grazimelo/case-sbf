import logging
import os

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../config", config_name="config")
def predict(cfg) -> None:
    # Endpoint
    ENDPOINT = cfg.app.endpoint
    # Input data
    INPUT_DATA = os.path.join(cfg.paths.inter, cfg.datasets.predict_data)
    # Output data
    OUTPUT_DATA = os.path.join(cfg.paths.output, cfg.datasets.output_data)
    VARIAVEIS_CATEGORICAS = [
        "polyuria",
        "perda_de_peso_repentina",
        "genero",
        "polyphagia",
        "polydipsia",
        "paresia_parcial",
    ]
    VARIAVEIS_NUMERICAS = ["idade"]
    try:
        X_predict = pd.read_parquet(INPUT_DATA)
        X_predict = X_predict[VARIAVEIS_CATEGORICAS + VARIAVEIS_NUMERICAS]
        logger.info("Input Data for prediction read successfully")
    except Exception as error:
        logger.error("Input data for prediction could not be read", exc_info=True)
        raise error
    # List to save each json generated
    list_predictions = []
    # Transforming records of dataframe into list of dictionaries
    X_predict_list = X_predict.to_dict(orient="records")
    # For each item of the list We will make a request
    try:
        for records in X_predict_list:
            res = requests.get(ENDPOINT, json=records)
            list_predictions.append(res.json())
        logger.info("Predictions were made successfully")
    except Exception as error:
        logger.error("Could not predict with data passed", exc_info=True)
        raise error
    # Exporting
    try:
        df_response = pd.DataFrame.from_records(list_predictions)
        df_response.to_csv(OUTPUT_DATA, index=False)
        logger.info("Predictions were exported successfully")
    except Exception as error:
        logger.error("Could export predictions", exc_info=True)
        raise error


if __name__ == "__main__":
    predict()
