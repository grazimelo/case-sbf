
  
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2_contingency
from boruta import BorutaPy


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

def point_biserial(df, y, num_columns = None, significancia=0.05):
    '''
    Perform feature selection based on correlation test.

            Parameters:
                    df (pandas.dataframe): A dataframe containing all features and target
                    num_columns (list): A list containing all categorical features. If empty list, the function tries to infer the categorical columns itself
                    y (string): A string indicating the target.

            Returns:
                    columns_remove_pb (list): 

    '''
    correlation = []
    p_values = []
    results = []
    
    
    if num_columns:
        num_columns = num_columns
    else:
        num_columns = df.select_dtypes(include=['int','float', 'int32', 'float64']).columns.tolist()
def boruta_selector(df, y=None):
    Y = df[y]
    df = df.drop(y,axis=1)
    num_feat = df.select_dtypes(include=['int','float']).columns.tolist()
    cat_feat = df.select_dtypes(include=['object']).columns.tolist()
    pipe_num_tree = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
    pipe_cat_tree = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
    preprocessor_tree = ColumnTransformer( transformers = [('num_preprocessor',pipe_num_tree, num_feat), ('cat_preprocessor', pipe_cat_tree, cat_feat)])
    RF  = Pipeline(steps = [('preprocessor_rf', preprocessor_tree),('model_rf',RandomForestClassifier(random_state = 123 ,max_depth =5))])
    X = preprocessor_tree.fit_transform(df)
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    
    # Criando o boruta
    feat_selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = 100) # 500 iterações até convergir
    feat_selector.fit(X,Y)
    # Terceiro filtro com as features selecionadas pelo boruta
    cols_drop_boruta= [not x for x in feat_selector.support_.tolist()] # apenas invertendo o vetor de true/false
    cols_drop_boruta= df.loc[:,cols_drop_boruta].columns.tolist()
    return cols_drop_boruta


# Função Chi2
def chi_squared(df, y, cols = None):
    pvalues = []
    logs = []
    chi2_list = []
    if cols == None:
        cat_columns = df.select_dtypes(['object']).columns.tolist()
    else:
        cat_columns = cols
    for cat in cat_columns:
        table = pd.crosstab(df[cat], df[y])
        if not table[table < 5 ].count().any():
            table = pd.crosstab(df[cat], df[y])
            chi2, p, dof, expected = chi2_contingency(table.values) # Função que realiza o teste

            chi2_list.append(chi2)
            pvalues.append(p)
        else:
            logs.append("A coluna {} não pode ser avaliada. ".format(cat))
            chi2_list.append(np.nan)
            pvalues.append(np.nan)   
    chi2_df = pd.DataFrame({"column":cat_columns, 'p-value':pvalues,'chi2_value':chi2_list})
    return  chi2_df, logs