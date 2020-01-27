#esse scrip faz o tratamento da meta-base retirando as colunas referentes às métricas f1m e auc dos algoritmos,
#retirando a coluna do algoritmo RF, criando o atributo "Class", que corresponde ao algortimo com melhor acurácia
# para um dado objeto e excluindo as colunas referentes à métrica acc (depois de atribuidas as classes). Além disso,
#as amostras com rótulo KNN foram duplicadas e algumas amostras com rótulo SVM foram removidas




import pandas as pd

#carrega meta-base e retira as colunas referentes as medidas acc e auc
data = pd.read_csv("/home/manuela/Documents/meta-base.csv", encoding = "UTF-8")
data = data.drop(["ANN.f1m", "ANN.auc", "C4.5.f1m", "C4.5.auc", "kNN.f1m", "kNN.auc", "SVM.f1m", "SVM.auc", "RF.acc", "RF.f1m", "RF.auc"],axis = 1)

#salva o arquivo da meta-base sem as medidas acc e auc pra poder usar se precisar
data.to_csv("meta-base_f1m.csv")


count = 0
maior = 0
ANN_samples = 0
SVM_samples = 0
C45_samples = 0
kNN_samples = 0

#pra cada linha, pega o maior valor entre as atributos correspondentes às
#acurácias e cria uma nova coluna chamada "classe" que recebe o algortimo que obteve essa acurácia
for i in data.index:
    maior = max(data.loc[i, ['ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc']])
    for j in ['ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc']:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data = data.drop(i, axis = 0)
            count = 0
            break
    count = 0


#retira algumas (60) amostras com rótulo SVM
for i in data.index:
    if (data.loc[i, "Class"] == 'SVM.acc'):
        SVM_samples += 1
        if SVM_samples > 60:
            data = data.drop(i, axis = 0)

#duplica algumas (39) amostras com rótulo kNN
for i in data.index:
    if (data.loc[i, "Class"] == 'kNN.acc') & (kNN_samples < 39):
            data = data.append(data.loc[i])
            kNN_samples += 1



#retira as colunas referentes ao score f1m nos diferentes algoritmos (já que ela já foi usada para rotulaçao)
data = data.drop(['ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc'], axis =1)

#exporta a base modificada
data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
