from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
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
  for to_drop in comb_to_drop[-1:]:
    data = raw.drop(list(to_drop), axis=1)
    considered_classes = ALL_ACC_SET - set(to_drop)
    data = preprocess_default(data, list( considered_classes ) )
    rfc = RandomForestClassifier(n_estimators=200)
    X = data.drop('Class', axis=1)
    y = data['Class']
    # 
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits)
    acc = 0
    classifiers_list.append([])
    mat_list.append([])
    for train_index, test_index in skf.split(X, y):
      rfc.fit(
        X.loc[train_index, :], data.loc[train_index ,'Class']
      )
      y_pred = rfc.predict(X.loc[test_index, :], )
      print("Considered classes : ", considered_classes)
      new_conf_mat = confusion_matrix(y_true=y.loc[test_index], y_pred=y_pred)
      new_acc = accuracy_score(y_true=y.loc[test_index], y_pred=y_pred)
      acc += new_acc
      print(f"accuracy: {new_acc}")
      print("precision: ", 
        precision_score(y_true=y.loc[test_index], y_pred=y_pred, average='macro', zero_division=0)
      )
      print("recall: ", 
        recall_score(y_true=y.loc[test_index], y_pred=y_pred, average='macro', zero_division=0)
      )
      print("f1_score: ", 
        f1_score(y_true=y.loc[test_index], y_pred=y_pred, average='macro', zero_division=0)
      )
      print("precision_recall_fscore_support : ")
      print(*precision_recall_fscore_support(y_true=y.loc[test_index], y_pred=y_pred, zero_division=0), sep= '\n')
      mat_list[-1].append( 
        new_conf_mat
       )
      # print(f"confusion_matrix:\n{mat_list[-1][-1]}")  
      classifiers_list[-1].append(rfc)      
      print("------------------------")
    print("mean(acc) = ", acc / n_splits)
    acc_mean_list.append(acc / n_splits)
