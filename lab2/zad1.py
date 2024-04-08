import pandas as pd

missing_values = ["-", "NA"]
df = pd.read_csv("lab2/iris_with_errors.csv", na_values=missing_values)

# print(df.values[0:1, :]) # [wiersz, rzad]
# print(df.head())

print(df.isnull().sum())

mediana_sepal_length = df["sepal.length"].median()
mediana_sepal_width = df["sepal.width"].median()
mediana_petal_length = df["petal.length"].median()
mediana_petal_width = df["petal.width"].median()

columns = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]

for column in columns:
    if column == "variety":
        i = 0
        for row1 in df[column]:
            if row1 != "Setosa" and row1 != "Versicolor" and row1 != "Virginica":
                print("Błędna wartość w wierszu: ", i, " w kolumnie: ", column, " wartość: ", row1)
                varity = ["Setosa", "Versicolor", "Virginica"]
                set = 0
                ver = 0
                vir = 0
                for var in varity:
                    x = 0
                    for litera in row1:
                        if litera == var[x]:
                            if var == "Setosa":
                                set += 1
                            if var == "Versicolor":
                                ver += 1
                            if var == "Virginica":
                                vir += 1
                        if x != len(litera):
                            x += 1 
                result = max(set, ver, vir)
                if result == set:
                    print("Poprawna wartość: Setosa")
                    df.loc[i, column] = "Setosa"
                elif result == ver:
                    print("Poprawna wartość: Versicolor")
                    df.loc[i, column] = "Versicolor"
                elif result == vir:
                    print("Poprawna wartość: Virginica")
                    df.loc[i, column] = "Virginica"
            i += 1
    else:
        median = df[column].median()
        i = 0
        for row in df[column]:
            if row <= 0 or 15 < row:
                df.loc[i, column] = median
            i += 1


df["sepal.length"] = df["sepal.length"].fillna(mediana_sepal_length)
df["sepal.width"] = df["sepal.width"].fillna(mediana_sepal_width)
df["petal.width"] = df["petal.width"].fillna(mediana_petal_width)

df.to_csv("lab2/iris_without_errors.csv", index=False)