import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("lab3/iris1.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285810)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming train_inputs, train_classes, test_inputs, test_classes are defined

k_values = [3, 5, 11]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)
    knn_predictions = knn.predict(test_inputs)
    print(f"k-NN (k={k}) accuracy: {accuracy_score(test_classes, knn_predictions)*100}%")
    print("Confusion Matrix:")
    print(confusion_matrix(test_classes, knn_predictions))

gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)
gnb_predictions = gnb.predict(test_inputs)
print(f"Naive Bayes accuracy: {accuracy_score(test_classes, gnb_predictions)*100}%")
print("Confusion Matrix:")
print(confusion_matrix(test_classes, gnb_predictions))

# c) w moim przypadku nalepiej sprawdzily sie knn 5 i 11 (~96% skutecznosci) a pozniej cala reszta(~93%)