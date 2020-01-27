
import pandas as pd

#carrega meta-base e retira as colunas referentes as medidas acc e auc
data = pd.read_csv("/home/manuela/Documents/meta-base.csv", encoding = "UTF-8")
data = data.drop(["ANN.acc", "ANN.auc", "C4.5.acc", "C4.5.auc", "kNN.acc", "kNN.auc", "SVM.acc", "SVM.auc", "RF.acc", "RF.auc"],axis = 1)

#salvei o arquivo da meta-base sem as medidas acc e auc pra poder usar se precisar
data.to_csv("meta-base_f1m.csv")

algoritm = ''
count = 0
maior = 0
RF_samples = 0
kNN_samples = 0
k = 0
#pra cada linha, peguei o maior valor entre os desempenhos e criei uma nova coluna chamada "classe"
#que recebe o algortimo correspondente a esse maior desempenho
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


for i in data.index:
    if (data.loc[i, "Class"] == 'RF.f1m'):
        RF_samples += 1
        if RF_samples > 60:
            data = data.drop(i, axis = 0)

for i in data.index:
    if (data.loc[i, "Class"] == 'kNN.f1m') & (kNN_samples < 30):
            data = data.append(data.loc[i])
            kNN_samples += 1

print(data.duplicated())

data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
