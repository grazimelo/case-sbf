import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def _normalizar_caracteres_aux(df: DataFrame) -> DataFrame:
    """Torna os caracteres de uma string minúsculos.

    Args:
        lista_variaveis_categoricas (list): Lista com as variáveis categóricas (strings)
                                             para serem transformadas.
        df (DataFrame): Dataframe para ser analisado.

    Returns:
        df: Dataframe transformado.
    """
    for col in df.columns:
        df[col] = df[col].str.lower()
    return df


def treinar_modelo(X: DataFrame, y: DataFrame) -> Pipeline:
    """_summary_

    Args:
        X (DataFrame): _description_
        y (DataFrame): _description_

    Returns:
        Pipeline: _description_
    """

    variaveis_categoricas = [
        "polyuria",
        "perda_de_peso_repentina",
        "genero",
        "polyphagia",
        "polydipsia",
        "paresia_parcial",
    ]
    variaveis_numericas = ["idade"]
    dt = DecisionTreeClassifier(random_state=123, class_weight="balanced")

    char_normalizer = FunctionTransformer(_normalizar_caracteres_aux)
    # Pipeline numérico será tratado imputando a mediana
    pipeline_numerico = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    # Pipeline categórico será tratado com imputação através da moda e one hot encoder
    pipeline_categorico = Pipeline(
        steps=[
            ("char_normalizer", char_normalizer),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    # Column Transformer para unir os dois pipelines
    preprocess_pipeline = ColumnTransformer(
        [
            ("pipeline_categorico", pipeline_categorico, variaveis_categoricas),
            ("pipeline_numerico", pipeline_numerico, variaveis_numericas),
        ]
    )
    modelo_pipe = Pipeline(
        steps=[("preprocessor", preprocess_pipeline), ("model_dt", dt)]
    )
    cv = RepeatedStratifiedKFold(random_state=123, n_splits=10, n_repeats=10)

    def optimize_decision_tree(trial):
        "Optmization of Decision Tree Classifier"
        # Grid de parametros otimizados
        params = {
            "model_dt__max_depth": trial.suggest_int("model_dt__max_depth", 2, 11),
            "model_dt__min_samples_leaf": trial.suggest_int(
                "model_dt__min_samples_leaf", 2, 33
            ),
            "model_dt__min_samples_split": trial.suggest_int(
                "model_dt__min_samples_split", 2, 33
            ),
        }
        earlyStop = 20
        modelo_pipe.set_params(**params)
        results = np.mean(
            cross_val_score(modelo_pipe, X, y, cv=cv, scoring="neg_log_loss")
        )
        print(f"LOG LOSS:{results}")
        return results

    sampler = TPESampler(seed=123)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(optimize_decision_tree, n_trials=25)
    modelo_pipe.set_params(**study.best_params)
    return modelo_pipe, study.best_params


def calcular_metricas_threshold(y_true, y_predict_proba_1):
    recall_score_list = []
    precision_score_list = []
    accuracy_score_list = []
    for thresh in np.linspace(0, 1, 100):
        recall_score_list.append(recall_score(y_true, y_predict_proba_1 >= thresh))
        precision_score_list.append(
            precision_score(y_true, y_predict_proba_1 >= thresh)
        )
        accuracy_score_list.append(accuracy_score(y_true, y_predict_proba_1 >= thresh))

    df_metrics = pd.DataFrame(
        {
            "threshold": np.linspace(0, 1, 100),
            "recall_score": recall_score_list,
            "precision_score": precision_score_list,
            "accuracy_score": accuracy_score_list,
        }
    )
    return df_metrics
