from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv("lab4/diabetes.csv")
# print(df.values)

X = df.iloc[:, :-1].values  # wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6,3), random_state=8, activation='relu',max_iter=500)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,3,3), random_state=8, activation='relu',max_iter=500)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=13, activation='relu',max_iter=500)
# clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12,12,), random_state=13, activation='tanh',max_iter=500)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12,12,), random_state=42, activation='relu',max_iter=500)


clf.fit(X_train, y_train)
  
predicted = clf.predict(X_test)

print(predicted)

print('The accuracy of the Multi-layer Perceptron is:', metrics.accuracy_score(predicted, y_test))

print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, predicted))