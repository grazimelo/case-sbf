

import seaborn as sns
import matplotlib.pyplot as plt

cmap = sns.diverging_palette(0,100,74,39,19,25, center='light', as_cmap=True) #heatmap

def plot_correlação(lista_de_variaveis, df):
    plt.figure(figsize=(20,4))
    corrmat = df.astype('int').loc[:,lista_de_variaveis].corr(method='pearson')
    sns.heatmap([corrmat['target']], xticklabels = corrmat.index,
                annot=True, fmt='.2f', annot_kws={'size': 14},
                cbar=False, center=0,cmap=cmap)
    plt.title('Correlação pelo método de Person')
    plt.tight_layout()
    plt.show()