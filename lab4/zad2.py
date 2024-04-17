from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=13)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=8)

# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# predicted = clf.predict(X_test)
# print(predicted)
# te liczby to predykcje dla irysow, 1- setosa, 2- versicolor, 3- virginica

# print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(predicted,y_test))


# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=13)

# Construct the neural network model
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=8)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=8)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=8)
# NAJLEPIEJ WYSZLA JEDNA 3 NEURONOWA WARSTWA UKRYTA

# Train the model
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predicted = clf.predict(X_test)

# Print the predictions
print(predicted)

# Print the accuracy of the model
print('The accuracy of the Multi-layer Perceptron is:', metrics.accuracy_score(predicted, y_test))