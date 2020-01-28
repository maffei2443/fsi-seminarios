"""
  Ao ser executado, dropa as colunas especificadas.

"""

import getopt
import numpy as np
import pandas as pd
import pandas

def drop_draw(data: pandas.DataFrame , columns = ['ANN.acc', 'kNN.acc', 'SVM.acc', 'RF.acc']):
  """
    Recebe um dataframe e dropa linhas cujo valor máximo nas colunas especificadas
    apareça mais de uma vez nelas.
  """
  for i in data.index:
    maior = np.max( max(data.loc[i, columns]) )
    count = 0
    for j in columns:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data.drop(i, axis = 0, inplace=True)
            count = 0
            break
  return data

def preprocess_default(path = 'meta-base.csv', columns = None):
  """
    Pre-procesamento padrao para problemas de classificacao.
    Consistem em:
      - dropar amostras cujo valor maximal da acurácia não é máximo
      - remover colunas com desempenho relativo à auc e f1-measure
  """
  data = drop_draw(pd.read_csv(path))
  # Remove colunas com medidas auc e f1m
  data.drop(data.filter(regex='(.*?\.auc|.*?\.f1m)').columns, axis=1, inplace=True)
  return data