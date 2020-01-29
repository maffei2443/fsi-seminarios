from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from aux import preprocess_default, ALL_ACC, ALL_ACC_SET
from itertools import combinations
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
  comb_to_drop = list(combinations(ALL_ACC, r=3))
  raw = pd.read_csv('meta-base.csv')
  
  classifiers_list = []
  mat_list = []
  acc_mean_list = []
  for to_drop in comb_to_drop:
    data = raw.drop(list(to_drop), axis=1)
    considered_classes = ALL_ACC_SET - set(to_drop)
    data = preprocess_default(data, list( considered_classes ) )
    rfc = RandomForestClassifier(n_estimators=200)
    X = data.drop('Class', axis=1)
    y = data['Class']
    # 
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    acc = 0
    acc_mat = None
    classifiers_list.append([])
    mat_list.append([])
    for train_index, test_index in skf.split(X, y):
      rfc.fit(
        X.loc[train_index, :], data.loc[train_index ,'Class']
      )
      y_pred = rfc.predict(X.loc[test_index, :], )
      print("Considered classes : ", considered_classes)
      new_acc = accuracy_score(y_true=y.loc[test_index], y_pred=y_pred)
      new_conf_mat = confusion_matrix(y_true=y.loc[test_index], y_pred=y_pred)
      acc += new_acc
      acc_mat = acc_mat + new_conf_mat if not isinstance(acc_mat, type(None))   else new_conf_mat
      print(f"Acc: {new_acc}")
      # print(f"precision: {precision_score(y_true=y.loc[test_index], y_pred=y_pred)}")
      # print(f"recall: {recall_score(y_true=y.loc[test_index], y_pred=y_pred)}")
      # print(f"f1: {f1_score(y_true=y.loc[test_index], y_pred=y_pred)}")
      mat_list[-1].append( 
        confusion_matrix(y_true=y.loc[test_index], y_pred=y_pred)
       )
      print(f"confusion_matrix:\n {mat_list[-1][-1]}")  
      classifiers_list[-1].append(rfc)      
      print("------------------------")
    print("mean(acc) = ", acc / n_splits)
    acc_mean_list.append(acc / n_splits)
    print("mean(Matriz de confusao):\n ", acc_mat / n_splits)
