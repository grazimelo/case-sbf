import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

cmap = sns.diverging_palette(0, 100, 74, 39, 19, 25, center="light", as_cmap=True)


def plotar_correlacao(
    lista_de_variaveis: list, df: DataFrame, method="pearson"
) -> None:
    """Plota correlação do target com as features"""
    plt.figure(figsize=(20, 4))
    corrmat = df.astype("int").loc[:, lista_de_variaveis].corr(method=method)
    sns.heatmap(
        [corrmat["target"]],
        xticklabels=corrmat.index,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
        cbar=False,
        center=0,
        cmap=cmap,
    )
    plt.title(f"Correlação pelo método de {method}")
    plt.tight_layout()
    plt.show()
