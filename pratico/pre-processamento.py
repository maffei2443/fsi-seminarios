
import pandas as pd


data = pd.read_csv("meta-base.csv", encoding = "UTF-8")

data = data.drop([
  "ANN.acc", "ANN.auc", "C4.5.acc", 
  "C4.5.auc", "kNN.acc", "kNN.auc",
  "SVM.acc", "SVM.auc", "RF.acc", "RF.auc"
  ],axis = 1)

data.to_csv("meta-base_f1m.csv")

algoritm = ''
count = 0
remove = []
maior = 0

for i in data.index:
    maior = max(data.loc[i, ['ANN.f1m', 'C4.5.f1m', 'kNN.f1m', 'SVM.f1m', 'RF.f1m']])
    for j in ['ANN.f1m', 'C4.5.f1m', 'kNN.f1m', 'SVM.f1m', 'RF.f1m']:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data = data.drop(i, axis = 0)
            count = 0
            break
    count = 0


data = data.drop(['ANN.f1m', 'C4.5.f1m', 'kNN.f1m', 'SVM.f1m', 'RF.f1m'], axis =1)

print(data)

data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
