import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from boruta import BorutaPy
from pandas import DataFrame
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def point_biserial(df, y, num_columns=None, significancia=0.05) -> tuple:
    """
    Realiza o teste da correlação point-biserial entre features numéricas e uma categórica binária.

    Args:
        df (DataFrame): Data frame para ser analisado.
        num_columns (list): Lista de colunas numéricas.
        y (string): String como  nome da coluna categórica binária.

    Returns:
            pb_df, columns_remove_pb (tuple): Dataframe com os resultados do teste de hipótese
                                              e lista de colunas para remover.

    """
    correlation = []
    p_values = []
    results = []

    if num_columns:
        num_columns = num_columns
    else:
        num_columns = df.select_dtypes(
            include=["int", "float", "int32", "float64"]
        ).columns.tolist()

    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
        correlation_aux, p_value_aux = pointbiserialr(df[col], df[y])
        correlation.append(correlation_aux)
        p_values.append(p_value_aux)

        if p_value_aux <= significancia:
            results.append("Reject H0")
        else:
            results.append("Accept H0")

    pb_df = pd.DataFrame(
        {
            "column": num_columns,
            "correlation": correlation,
            "p_value": p_values,
            "result": results,
        }
    )
    columns_remove_pb = pb_df.loc[pb_df["result"] == "Accept H0"][
        "column"
    ].values.tolist()

    return pb_df, columns_remove_pb


def boruta_selector(df: DataFrame, y: str = None) -> list:
    """Realiza seleção de feautres pelo algoritmo Boruta.

    Args:
        df (DataFrame): Dataframe para ser analisado.
        y (str): Nome da variável target.

    Returns:
        cols_drop_boruta: Lista com as colunas que devem ser removidas
    """
    Y = df[y]
    df = df.drop(y, axis=1)
    num_feat = df.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_feat = df.select_dtypes(include=["object"]).columns.tolist()
    pipe_num_tree = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    pipe_cat_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_transformer", OrdinalEncoder()),
        ]
    )
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num_preprocessor", pipe_num_tree, num_feat),
            ("cat_preprocessor", pipe_cat_tree, cat_feat),
        ]
    )
    RF = Pipeline(
        steps=[
            ("preprocessor_rf", preprocessor_tree),
            ("model_rf", RandomForestClassifier(random_state=123, max_depth=5)),
        ]
    )
    X = preprocessor_tree.fit_transform(df)
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    # Criando o boruta
    feat_selector = BorutaPy(
        rf, n_estimators="auto", random_state=123, max_iter=100
    )  # 500 iterações até convergir
    feat_selector.fit(X, Y)
    # Terceiro filtro com as features selecionadas pelo boruta
    cols_drop_boruta = [
        not x for x in feat_selector.support_.tolist()
    ]  # apenas invertendo o vetor de true/false
    cols_drop_boruta = df.loc[:, cols_drop_boruta].columns.tolist()
    return cols_drop_boruta


# Função Chi2
def chi_squared(df: DataFrame, y: str, cols: list = None) -> tuple:
    """Realiza o teste de qui-quadrado entre variáveis categóricas e target categórica.

    Args:
        df (DataFrame): Data frame com as variáveis para serem analisadas.
        y (string): Target categórico.
        cols (list): Lista de colunas categóricas.

    Returns:
        chi2_df, logs: Data frame com os resultados
                       e logs indicando se alguma coluna não pôde passar pelo teste.
    """
    pvalues = []
    logs = []
    chi2_list = []
    if cols == None:
        cat_columns = df.select_dtypes(["object"]).columns.tolist()
    else:
        cat_columns = cols
    for cat in cat_columns:
        table = pd.crosstab(df[cat], df[y])
        if not table[table < 5].count().any():
            table = pd.crosstab(df[cat], df[y])
            chi2, p, dof, expected = chi2_contingency(
                table.values
            )  # Função que realiza o teste

            chi2_list.append(chi2)
            pvalues.append(p)
        else:
            logs.append("A coluna {} não pode ser avaliada. ".format(cat))
            chi2_list.append(np.nan)
            pvalues.append(np.nan)
    chi2_df = pd.DataFrame(
        {"column": cat_columns, "p-value": pvalues, "chi2_value": chi2_list}
    )
    return chi2_df, logs
