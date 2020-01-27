#nesse script é usada a validação cruzada com 10 folds para avaliar os algortimos RandomForest, SVC e KNN
#sobre a meta-base de dados normalizada


import pandas as pd
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

scores = []

#carrega a base com as features e as classes
data = pd.read_csv("data_classes.csv")

#divide a base em atributos e as classes
X = data.drop("Class", axis = 1)
y = data["Class"]

#printa a proporção de objetos de cada classe
print(collections.Counter(y))

#normaliza os atributos
scaler = StandardScaler()
X = scaler.fit_transform(X)

#classificadores que serão testados
rfc = RandomForestClassifier(n_estimators = 200)
knn = KNeighborsClassifier(n_neighbors = 5)
clf = svm.SVC()

#avaliação dos algoritmos usando validação cruzada com 10 folds
scores.append(cross_val_score(rfc, X, y, cv=10).mean())
scores.append(cross_val_score(knn, X, y, cv=10).mean())
scores.append(cross_val_score(clf, X, y, cv=10).mean())
print(scores)
