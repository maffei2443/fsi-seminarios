"""
  Ao ser executado, dropa as colunas especificadas.

"""

import getopt
import numpy as np
import pandas as pd
import pandas
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, \
  precision_score,recall_score, f1_score



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

def print_counter(data: pandas.DataFrame):
  for i in Counter(data).items():
    print(*i, sep=' -> ')

def show_metrics(y_true, y_pred):
  print(f"accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}")
  print("precision: ", 
    precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
  )
  print("recall: ", 
    recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
  )
  print("f1_score: ", 
    f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
  )
  print("precision_recall_fscore_support : ")
  print(
    "\t", *precision_recall_fscore_support(y_true=y_true,
      y_pred=y_pred, zero_division=0), sep= '\n\t')

def show_most_important_features(X: pd.DataFrame, features_list: list):
  for ft_lis in features_list:
      x=[]
      y=[]
      for i, j in sorted(
        zip(X.columns[1:], ft_lis), 
        key = lambda x : x[1], reverse=True)[:7]:
          x.append(i)
          y.append(j)
      df = pd.DataFrame({'lab':x, 'val':y})
      ax = df.plot.barh(x='lab', y='val', rot=0)
