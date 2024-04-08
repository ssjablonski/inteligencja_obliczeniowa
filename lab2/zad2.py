from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="variety")
print(X.head())

pca_iris = PCA(n_components=3).fit(iris.data)
# print("pca", pca_iris)
print("war",pca_iris.explained_variance_ratio_)
# print(pca_iris.components_)
# print(pca_iris.transform(iris.data))

cumulative_variance_ratio = np.cumsum(pca_iris.explained_variance_ratio_)

num_components = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1

print(f"Ilość kolumn aby zachowac 95% wariancji: {num_components}")

if num_components == 2:
    plt.scatter(pca_iris.transform(iris.data)[:, 0], pca_iris.transform(iris.data)[:, 1], c=iris.target)
    # te transfomy biora wszystkie wiersze z pierwszej kolumny i drugiej kolumny
    plt.title('PCA 2D')
    plt.show()
elif num_components == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_iris.transform(iris.data)[:, 0], pca_iris.transform(iris.data)[:, 1], pca_iris.transform(iris.data)[:, 2], c=iris.target)
    # tu tak samo tylko 3 a nie 2
    ax.set_title('PCA 3D')
    plt.show()
else:
    print("Invalid number of components.")