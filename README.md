diabetes_app
==============================
Este conjunto de dados contém os dados de sinais e sintomas de recém-diabéticos ou pacientes diabéticos.
Project Organization
------------
    ├── LICENSE
    ├── README.md
    ├── app.py -> Aplicativo Flask com o modelo.
    ├── config
    │   └── config_model.yaml
    |   └── app
    |       └── app_config.yaml
    |   └──  datasets
    |        └── datasets.yaml
    |   └── mlflow
    |        └── mlflow.yaml
    |   └── model
    |        └── model.yaml
    |   └── paths
    |        └── paths.model
    ├── data -> Pasta com todos os dados.
    │   ├── inter -> Pasta com os arquivos intermediários.
    │       └── diabetes_test.parquet -> Dados de teste.
    |       └── diabetes_train.parquet -> Dados de treino.
    |       └── diabetes_train_selected.parquet -> Dados com as variáveis selecionadas. 
    |
    │   ├── Ouput - > Pasta com as previsões do modelo. 
    │   │   └── predictions.csv -> Previsões do modelo
    │   └── raw -> Pasta com todos os arquivos fornecidos.
    │       └── diabetes_data.csv   
    ├── notebooks -> Pasta com os notebooks de desenvolvimento.
    │   ├── 1_data_cleaning.ipynb -> Limpeza dos dados
    │   ├── 2_eda.ipynb -> Análise exploratória
    |   ├── 3_selecao_de_features.ipynb - > Selecão de features
    │   └── 4_modelagem.ipynb ->  Modelagem
    ├── outputs -> Saída do modelo
    ├── src -> Dependências
    |    └── outputs
    |    └── predict.py -> Previsões
    |    └── train_model.py -> Modelo treinado
    ├── requirements.txt -> Requisitos para o projeto.
    ├── setup.py 
    └── utils -> Módulo com todas as funções e classes criadas para o projeto
        ├── _init_.py
        ├── data_cleaning.py
        └── eda.py
        ├── modelagem.py
        └── selecao_de_features.py
------------

1) Criar ambiente
    - Criar virtualenv
    - Ativar virtualenv
      git clone <https://github.com/grazimelo/case-sbf.git>
      cd case_sbf
      pip install -r requirements.txt
------------
2) Instanciar mlflow
    mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root file://$PWD/mlruns
------------
3) Treino/Retreino
    Copiar novo arquivo (em formato parquet) com nome diabetes_train.parquet para a pasta data/inter (o formato deve ser equivalente ao formato do arquivo atual)
    cd src
    python train_model.py
------------
4) Prever novos dados
    Para prever basta copiar um arquivo com o nome diabetes_test.parquet para a pasta data/inter
    python app.py (sobe o app flask)
    cd src
    python predict.py

    O arquivo com as previsões irão para a pasta data/output
------------
