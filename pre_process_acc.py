#esse scrip faz o tratamento da meta-base retirando as colunas referentes às métricas f1m e auc dos algoritmos,
#criando o atributo "Class", que corresponde ao algortimo com melhor acurácia para um dado objeto e excluindo
#as colunas referentes à métrica acc (depois de atribuidas as classes).
#Distribuição das classes na base: {'RF.acc': 155, 'SVM.acc': 95, 'ANN.acc': 64, 'C4.5.acc': 53, 'kNN.acc': 18}
#Retirando os sinais de comentário das linhas 41-52 obtemos a base balanceada onde foram retirados 95
#objetos com rótulo RF e duplicados os objetos com rótulo kNN: {'SVM.acc': 95, 'ANN.acc': 64, 'RF.acc': 60, 'C4.5.acc': 53, 'kNN.acc': 36}



import pandas as pd

#carrega meta-base e retira as colunas referentes as medidas acc e auc
data = pd.read_csv("/home/manuela/Documents/meta-base.csv", encoding = "UTF-8")
data = data.drop(["ANN.f1m", "ANN.auc", "C4.5.f1m", "C4.5.auc", "kNN.f1m", "kNN.auc", "SVM.f1m", "SVM.auc", "RF.f1m", "RF.auc"],axis = 1)

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
    maior = max(data.loc[i, ['RF.acc','ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc']])
    for j in ['RF.acc', 'ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc']:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data = data.drop(i, axis = 0)
            count = 0
            break
    count = 0


"""#retira alguns objetos com rótulo RF
for i in data.index:
    if (data.loc[i, "Class"] == 'RF.acc'):
        RF_samples += 1
        if RF_samples > 60:
            data = data.drop(i, axis = 0)

#duplica algumas (30) amostras com rótulo kNN
for i in data.index:
    if (data.loc[i, "Class"] == 'kNN.acc') & (kNN_samples < 30):
            data = data.append(data.loc[i])
            kNN_samples += 1"""



#retira as colunas referentes ao score f1m nos diferentes algoritmos (já que ela já foi usada para rotulaçao)
data = data.drop(['RF.acc','ANN.acc', 'C4.5.acc', 'kNN.acc', 'SVM.acc'], axis =1)


#exporta a base modificada
data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
