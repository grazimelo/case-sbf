from pandas import DataFrame


def remover_duplicatas(df: DataFrame, drop_if_found=True) -> tuple:
    """Remove linhas e colunas duplicadas no dataframe.

    Args:
        df (DataFrame): Um dataframe para ser analisado.
        drop_if_found (bool): Flag indicando se é pra devolver o dataframe ou só analisar.

    Returns:
        df_T, list_duplicated_columns: Dataframe sem linhas e colunas duplicadas
                                       e lista com as colunas duplicadas
    """

    # Verificando duplicatas colunas
    print(
        f"Existem {df.T.duplicated().sum()} colunas duplicadas e {df.duplicated().sum()} linhas duplicadas"
    )
    # Verificando duplicatas nas linhas
    if drop_if_found == True:
        print("Removendo...")
        df = df.drop_duplicates()
        df_T = df.T
        df_T.drop_duplicates(inplace=True)
        list_duplicated_columns = df_T[df_T.duplicated(keep=False)].index.tolist()
        print("Colunas duplicadas:")
        print(list_duplicated_columns)
        return df_T.T, list_duplicated_columns
    else:
        list_duplicated_columns = df.T[df.T.duplicated(keep=False)].index.tolist()
        return list_duplicated_columns


def remover_colunas_constantes(df: DataFrame) -> tuple:
    """Remove colunas constantes no dataframe.

    Args:
        df (DataFrame): Dataframe para ser analisado.

    Returns:
        df, const_cols: Dataframe sem colunas constantes
                        e lista com as colunas constantes encontradas
    """
    const_cols = []
    for i in df.columns:
        if len(df[i].unique()) == 1:
            df.drop(i, axis=1, inplace=True)
            const_cols.append(i)
    return df, const_cols


def normalizar_caracteres(
    lista_variaveis_categoricas: list, df: DataFrame
) -> DataFrame:
    """Torna os caracteres de uma string minúsculos.

    Args:
        lista_variaveis_categoricas (list): Lista com as variáveis categóricas (strings)
                                             para serem transformadas.
        df (DataFrame): Dataframe para ser analisado.

    Returns:
        df: Dataframe transformado.
    """
    for var_cat in lista_variaveis_categoricas:
        df[var_cat] = df[var_cat].str.lower()
    return df
