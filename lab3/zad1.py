# import pandas as pd
# from sklearn.model_selection import train_test_split


# df = pd.read_csv("lab3/iris1.csv")
# # print(df)

# #podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
# (train_set, test_set) = train_test_split(df.values, train_size=0.7,
# random_state=13)

# print(test_set)
# print(test_set.shape[0])

# train_inputs = train_set[:, 0:4]
# train_classes = train_set[:, 4]
# test_inputs = test_set[:, 0:4]
# test_classes = test_set[:, 4]

# print(train_inputs)
# print(train_classes)
# print(test_inputs)
# print(test_classes)

import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("lab3/iris1.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285810)

def classify_iris(sl, sw, pl, pw):
    if sl <= 6 and sw >= 3 and pl < 3 and pw < 1:
        return("Setosa")
    elif 3 <= pl <= 5 and 1 <= pw <= 1.7:
        return("Versicolor")
    else:
        return("Virginica")
    
good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1
    print(test_set[i])
print(good_predictions)
print(good_predictions/len*100, "%")

 

