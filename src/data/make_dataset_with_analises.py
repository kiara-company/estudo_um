# import numpy as np
import pandas as pd
from pycaret.classification import (
    compare_models,
    finalize_model,
    plot_model,
    predict_model,
    setup,
    tune_model,
)

df_teste = pd.read_csv("../../data/interin/test.csv")
df_treino = pd.read_csv("../../data/interin/train.csv")

df_treino.head()
df_treino.shape
len(df_treino["id"].unique())
df_treino.dtypes  # ver tipos de colunas
len(df_treino.dtypes)  # 21 tem a mais churn
df_treino.isna().sum()

df_treino["TotalCharges"] = pd.to_numeric(df_treino["TotalCharges"], errors="coerce")
df_treino.dtypes

setup_de_classificacao_treino = setup(
    data=df_treino, target="Churn", fold_strategy="kfold"
)

melhores_modelos = compare_models(n_select=5)
melhores_modelos[0]
type(melhores_modelos[0])

gbc_tunado = tune_model(melhores_modelos[0])
modelo_final = finalize_model(gbc_tunado)
modelo_final
plot_model(gbc_tunado)
plot_model(gbc_tunado, plot="confusion_matrix")
plot_model(gbc_tunado, plot="feature")

df_teste["TotalCharges"] = pd.to_numeric(df_teste["TotalCharges"], errors="coerce")
predicoes = predict_model(modelo_final, data=df_teste)
predicoes
predicoes.columns

len(predicoes["prediction_label"])  # 1409
pd.DataFrame({"id": df_teste["id"], "Churn": predicoes["prediction_label"]})

import plotly.express as px

fig = px.scatter(
    x=df_treino["tenure"],
    y=df_treino["TotalCharges"],
    color=df_treino["Churn"],
    template="presentation",
    opacity=0.5,
    facet_col=df_treino["Contract"],
    title="Cliente Churn by Permanência, Taxas e Tipo de Contrato",
    labels={"x": "Permanência do Cliente", "y": "Total Taxas $"},
)

fig.show()
# setup_de_classificacao_teste = setup(data=df_teste, target="Churn")
