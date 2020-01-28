#esse scrip faz o tratamento da meta-base retirando as colunas referentes às métricas f1m e auc dos algoritmos,
#retirando a coluna do algoritmo C4.5, criando o atributo "Class", que corresponde ao algortimo com melhor acurácia
# para um dado objeto e excluindo as colunas referentes à métrica acc (depois de atribuidas as classes).
#Distribuição das classes na base: {'RF.acc': 183, 'SVM.acc': 108, 'ANN.acc': 77, 'kNN.acc': 18}




import pandas as pd

#carrega meta-base e retira as colunas referentes as medidas acc e auc
data = pd.read_csv("/home/manuela/Documents/meta-base.csv", encoding = "UTF-8")
data = data.drop(["ANN.f1m", "ANN.auc", "C4.5.acc","C4.5.f1m", "C4.5.auc", "kNN.f1m", "kNN.auc", "SVM.f1m", "SVM.auc", "RF.f1m", "RF.auc"],axis = 1)

#salva o arquivo da meta-base sem as medidas acc e auc pra poder usar se precisar
data.to_csv("meta-base_f1m.csv")

count = 0
maior = 0
RF_samples = 0
kNN_samples = 0
k = 0

#pra cada linha, pega o maior valor entre os desempenhos e cria uma nova coluna chamada "classe"
#que recebe o algortimo correspondente a esse maior desempenho
for i in data.index:
    maior = max(data.loc[i, ['ANN.acc', 'kNN.acc', 'SVM.acc', 'RF.acc']])
    for j in ['ANN.acc', 'kNN.acc', 'SVM.acc', 'RF.acc']:
        if data.loc[i,j] == maior:
            count += 1
            data.loc[i, "Class"] = j
        if count > 1:
            data = data.drop(i, axis = 0)
            count = 0
            break
    count = 0


#retira as colunas referentes ao score f1m nos diferentes algoritmos (já que ela já foi usada para rotulaçao)
data = data.drop(['ANN.acc', 'kNN.acc', 'SVM.acc', 'RF.acc'], axis =1)


#exporta a base modificada
data.to_csv("data_classes.csv", index = False, encoding = "UTF-8")
