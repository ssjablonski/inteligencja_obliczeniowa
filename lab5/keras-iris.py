import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
# standardizuje cechy poprzez usunięcie średniej i skalowanie do jednostkowej wariancji.
# Standardowy wynik próbki x jest obliczany jako z = (x - u) / s, gdzie u to średnia z próbek treningowych, a s to odchylenie standardowe próbek treningowych.
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder()
#  to metoda przekształcania etykiet kategorialnych w binarne wektory, gdzie każda kategoria jest reprezentowana przez jeden bit ustawiony na 1, a reszta bitów ustawiona na 0. 
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
]) # 97.8%
# input_shape=(X_train.shape[1],) oznacza, że model oczekuje na wejściu tyle cech, ile kolumn ma X_train
# y_encoded.shape[1] to liczba klas, które zostały zakodowane na gorąco, czyli 3 dla zbioru Iris. Warstwa wyjściowa ma zatem 3 neurony, co odpowiada 3 możliwym klasom wynikowym.

# model = Sequential([
#     Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='sigmoid'),
#     Dense(y_encoded.shape[1], activation='softmax')
# ]) # 95.6%

# model = Sequential([
#     Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='tanh'),
#     Dense(y_encoded.shape[1], activation='softmax')
# ]) # 97.8%

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # 84%
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # 100%!


# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) # 100?
# model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy']) # 100?
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 100?
# model.compile(optimizer='adam', loss='poisson', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2 , batch_size=16)
# batch_size=16 oznacza, że model będzie aktualizowany po każdych 16 próbkach treningowych

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Dokładność na zbiorze treningowym i walidacyjnym generalnie wzrasta wraz z kolejnymi epokami, co jest pozytywnym znakiem, wskazującym na to, że model uczy się i adaptuje do danych. Dokładność treningowa osiągnęła bardzo wysoki poziom, zbliżając się do 100%, co sugeruje, że model bardzo dobrze nauczył się rozpoznawać wzorce w danych treningowych.

# proces ładowania danych, ich przetwarzania (standaryzacja, kodowanie), podziału na zestawy treningowe i testowe, budowy modelu sieci neuronowej,
# jego kompilacji, treningu, ewaluacji, i wizualizacji wyników, a także zapisu modelu i architektury. 