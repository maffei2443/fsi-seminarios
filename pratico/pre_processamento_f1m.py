
#esse scrip faz o tratamento da meta-base retirando as colunas referentes às métricas acc e auc,
#criando o atributo "Class", que corresponde ao algortimo com melhor score f1m para um dado objeto,
#excluindo as colunas referentes à métrica f1m (depois de atribuidas as classes) e, por fim, balanceando
#a base a partir da retirada de amostras com rótulo RF.f1m e duplicação de amostras com rótulo kNN.f1m.




import pandas as pd

#carrega meta-base e retira as colunas referentes as medidas acc e auc
data = pd.read_csv("/home/manuela/Documents/meta-base.csv", encoding = "UTF-8")
data = data.drop(["ANN.acc", "ANN.auc", "C4.5.acc", "C4.5.auc", "kNN.acc", "kNN.auc", "SVM.acc", "SVM.auc", "RF.acc", "RF.auc"],axis = 1)

#salva o arquivo da meta-base sem as medidas acc e auc pra poder usar se precisar
data.to_csv("meta-base_f1m.csv")

algoritm = ''
count = 0
maior = 0
RF_samples = 0
kNN_samples = 0
k = 0
#pra cada linha, pega o maior valor entre os desempenhos e cria uma nova coluna chamada "classe"
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


#retira as colunas referentes ao score f1m nos diferentes algoritmos (já que ela já foi usada para rotulaçao)
data = data.drop(['ANN.f1m', 'C4.5.f1m', 'kNN.f1m', 'SVM.f1m', 'RF.f1m'], axis =1)

#retira algumas (60) amostras com rótulo RF
for i in data.index:
    if (data.loc[i, "Class"] == 'RF.f1m'):
        RF_samples += 1
        if RF_samples > 60:
            data = data.drop(i, axis = 0)

#duplica algumas (30) amostras com rótulo kNN
for i in data.index:
    if (data.loc[i, "Class"] == 'kNN.f1m') & (kNN_samples < 30):
            data = data.append(data.loc[i])
            kNN_samples += 1

#exporta a base modificada
data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
