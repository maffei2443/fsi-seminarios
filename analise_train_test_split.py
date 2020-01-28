#nesse script dividimos a meta-base já balanceada em conjuntos de treinamento e teste de forma aleatória,
#onde teste contém 20% das amostras. Em seguida induzimos modelos usando os algoritmos RandomForest, KNN e SVC
#e avaliamos suas predições usando matrizes de confusão e relatórios de classificação.




import pandas as pd
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


#carrega a base com as features e as classes
data = pd.read_csv("data_classes.csv")

#divide as features e as classe
X = data.drop("Class", axis = 1)
y = data["Class"]

print(collections.Counter(y))
#separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print("Acurracy: ", format(accuracy), "\n")
print("Matriz de confusão: \n \n", confusion_matrix(pred_rfc, y_test), "\n")
print("Relatório de classiicação: \n \n", classification_report(pred_rfc, y_test), "\n")
print("---------------------------------------------")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
accuracy = accuracy_score(y_test, pred_knn)
print("Acurracy: ", format(accuracy), "\n")
print("Matriz de confusão: \n \n", confusion_matrix(pred_knn, y_test), "\n")
print("Relatório de classiicação: \n \n", classification_report(pred_knn, y_test), "\n")
print("---------------------------------------------")



"""clf = svm.SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred_clf)
print("Acurracy: ", format(accuracy), "\n")
print("Matriz de confusão: \n \n", confusion_matrix(pred_clf, y_test), "\n")
print("Relatório de classiicação: \n \n", classification_report(pred_clf, y_test), "\n")
print("---------------------------------------------")"""
