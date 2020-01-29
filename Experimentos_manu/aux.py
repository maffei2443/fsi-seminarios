"""
  Ao ser executado, dropa as colunas especificadas.

"""

import getopt
import numpy as np
import pandas as pd
import pandas
from collections import Counter



ALL_ACC = ['ANN.acc', 'kNN.acc', 'C4.5.acc', 'SVM.acc', 'RF.acc']
ALL_ACC_SET =  set(ALL_ACC)

def drop_draw_set_class(data: pandas.DataFrame , score_columns = ALL_ACC):
  """
    Recebe um dataframe e dropa linhas cujo valor máximo nas colunas especificadas
    apareça mais de uma vez nelas.
  """
  for i in data.index:
    maior = np.max( max(data.loc[i, score_columns]) )
    count = 0
    for j in score_columns:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data.drop(i, axis = 0, inplace=True)
            count = 0
            break
  data.drop(score_columns, axis=1, inplace=True)
  data.reset_index(inplace = True)
  return data

def drop_all_vs_rf(data: pandas.DataFrame , score_columns = ALL_ACC):
  """
    Recebe um dataframe e dropa linhas cujo valor máximo nas colunas especificadas
    apareça mais de uma vez nelas.
  """
  for i in data.index:
    maior = np.max( max(data.loc[i, score_columns]) )
    count = 0
    for j in score_columns:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j if j.startswith("RF") else "Other"
        if count > 1:
            data.drop(i, axis = 0, inplace=True)
            count = 0
            break
  data.drop(score_columns, axis=1, inplace=True)
  data.reset_index(inplace = True)
  return data


def preprocess_default(data: pandas.DataFrame, score_columns:list = []):
  """
    Pre-procesamento padrao para problemas de classificacao.
    Consistem em:
      - dropar amostras cujo valor maximal da acurácia não é máximo
      - remover colunas com desempenho relativo à auc e f1-measure
  """
  # Remove colunas com medidas auc e f1m
  data.drop(data.filter(regex='(.*?\.auc|.*?\.f1m)').columns, axis=1, inplace=True)

  # Dropar colunas sem vencedor na acuracia
  data = drop_draw_set_class(data, score_columns) if score_columns else drop_draw_set_class(data)
  return data





