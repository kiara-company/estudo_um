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

df_teste.head()
df_treino.head()
df_treino["TotalCharges"] = pd.to_numeric(df_treino["TotalCharges"], errors="coerce")
df_teste["TotalCharges"] = pd.to_numeric(df_teste["TotalCharges"], errors="coerce")

setup_de_classificacao_treino = setup(
    data=df_treino, target="Churn", fold_strategy="kfold"
)

melhores_modelos = compare_models(n_select=5)

gbc_tunado = tune_model(melhores_modelos[0])
modelo_final = finalize_model(gbc_tunado)

predicoes = predict_model(modelo_final, data=df_teste)

pd.DataFrame({"id": df_teste["id"], "Churn": predicoes["prediction_label"]})
