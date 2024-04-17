import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv("lab3/iris1.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285810)


train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)


fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(dtc, feature_names=df.columns[:4], class_names=list(np.unique(train_classes)), filled=True, ax=ax)
plt.show()

accuracy = dtc.score(test_inputs, test_classes)
print("Accuracy:", accuracy)

# predictions = dtc.predict(test_inputs)
# confusion_matrix = confusion_matrix(test_classes, predictions)
# print("Confusion Matrix:")
# print(confusion_matrix)


combined_inputs = np.concatenate((train_inputs, test_inputs), axis=0)
combined_classes = np.concatenate((train_classes, test_classes), axis=0)

combined_predictions = dtc.predict(combined_inputs)

combined_cm = confusion_matrix(combined_classes, combined_predictions, labels=np.unique(combined_classes))

print("Combined Confusion Matrix:")
print(combined_cm)

sns.heatmap(combined_cm, annot=True, fmt='d', xticklabels=np.unique(combined_classes), yticklabels=np.unique(combined_classes))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()