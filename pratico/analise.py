import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

#carrega a base com as features e as classes
data = pd.read_csv("data_classes.csv")

#divide as features e as classe
X = data.drop("Class", axis = 1)
y = data["Class"]

#separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(confusion_matrix(pred_rfc, y_test))
print(classification_report(pred_rfc, y_test))


mlpc = MLPClassifier(hidden_layer_sizes = (11,11,11), max_iter = 500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
print(confusion_matrix(pred_mlpc, y_test))
print(classification_report(pred_mlpc, y_test))



#treinamento
clf = svm.SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)
print(confusion_matrix(pred_clf, y_test))
print(classification_report(pred_clf, y_test))
