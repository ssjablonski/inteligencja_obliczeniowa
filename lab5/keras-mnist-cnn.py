import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
#
from keras.callbacks import ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
# zmienia kształt obrazów testowych na 4D tensor o wymiarach (liczba obrazów, wysokość obrazu, szerokość obrazu, liczba kanałów kolorów).
# Obrazy są również konwertowane na typ danych float32 i normalizowane do wartości między 0 a 1 poprzez podzielenie przez 255 (maksymalna wartość piksela w obrazie w skali szarości)
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
# konwertuje etykiety na wartości liczbowe zakodowane na gorąco
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # zapisuje oryginalne etykiety dla confusion matrix
# spowrotem konwertuje etykiety na wartości liczbowe z zakodowanych na gorąco

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # 32 filtry o wymiarach 3x3 i oczekuje obrazów wejściowych o wymiarach 28x28x1
    MaxPooling2D((2, 2)), # redukuje
    Flatten(), # spłaszcza dane wejściowe
    Dense(64, activation='relu'), # warstwa ukryta z 64 neuronami i relu
    Dense(10, activation='softmax') # warstwa wyjściowa z 10 neuronami i funkcją aktywacji softmax (10 bo tyle klas)
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# najwiekszy problem jest ze zgadnieciem liczby 5 (myli sie z 3 oraz 6) jak i 7 (myli z 2 i 3) oraz 8 (myli z 2 i 3)

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_loss', save_best_only=True, mode='auto')
#to sprawi zapistwanie modelu co epoke
# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()



# Krzywa dokładności treningowej jest wyższa niż krzywa straty treningowej. To typowy objaw przeuczenia. Model uczy się zbyt dobrze dopasowywać do danych treningowych, co prowadzi do pogorszenia wyników na danych walidacyjnych.
# Krzywa dokładności treningowej osiąga plateau i zaczyna spadać w późniejszych epokach. To kolejny objaw przeuczenia. Model "zapamiętuje" zbyt wiele szczegółów z danych treningowych, co powoduje, że traci zdolność do uogólniania na nowe dane.
# Krzywa straty walidacyjnej jest wyższa niż krzywa dokładności walidacyjnej. To kolejny objaw przeuczenia. Model lepiej radzi sobie z danymi treningowymi niż z danymi walidacyjnymi, co oznacza, że nie jest w stanie uogólniać na nowe dane.