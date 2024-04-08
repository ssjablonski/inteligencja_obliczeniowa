import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('lab2/iris1.csv')

min_max_scaler = MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(df[['sepal.length', 'sepal.width']])
data_min_max_scaled = pd.DataFrame(data_min_max_scaled, columns=['sepal.length', 'sepal.width'])
data_min_max_scaled['variety'] = df['variety']

standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(df[['sepal.length', 'sepal.width']])
data_standard_scaled = pd.DataFrame(data_standard_scaled, columns=['sepal.length', 'sepal.width'])
data_standard_scaled['variety'] = df['variety']

def plot_data(df, title):
    plt.figure(figsize=(8, 6))
    variety = df['variety'].unique()
    colors = ['red', 'green', 'blue']
    for i, spec in enumerate(variety):
        subset = df[df['variety'] == spec]
        plt.scatter(subset['sepal.length'], subset['sepal.width'], color=colors[i], label=spec)
    plt.title(title)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()

plot_data(df, 'Original Data')
plot_data(data_min_max_scaled, 'Min-Max Normalized Data')
plot_data(data_standard_scaled, 'Z-Score Scaled Data')

def plot_data_2(df, title):
    plt.figure(figsize=(8, 6))
    variety = df['variety'].unique()
    colors = ['red', 'green', 'blue']
    for i, spec in enumerate(variety):
        subset = df[df['variety'] == spec]
        plt.scatter(subset['petal.length'], subset['petal.width'], color=colors[i], label=spec)
    plt.title(title)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()

# plot_data_2(df, 'Original Data petal')


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# # Load the data
# df = pd.read_csv('lab2/iris1.csv')

# # Create a figure and a set of subplots
# fig, axs = plt.subplots(3)

# color_dict = {'Setosa': 'red', 'Versicolor': 'green', 'Virginica': 'blue'}
# df['color'] = df['varietyety'].map(color_dict)
# df.plot(kind='scatter', x='sepal.length', y='sepal.width', c='color', ax=axs[0], title='Original')

# # Plot original data
# df.plot(kind='scatter', x='sepal.length', y='sepal.width', c='varietyety', ax=axs[0], title='Original')

# # Normalize data using MinMaxScaler
# scaler = MinMaxScaler()
# df[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])
# df.plot(kind='scatter', x='sepal.length', y='sepal.width', c='varietyety', ax=axs[1], title='Normalized Min-Max')

# # Scale data using StandardScaler (Z-score)
# scaler = StandardScaler()
# df[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])
# df.plot(kind='scatter', x='sepal.length', y='sepal.width', c='varietyety', ax=axs[2], title='Scaled Z-score')

# # Show the plot
# plt.show()