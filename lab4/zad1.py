import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forwardPass(wiek, waga, wzrost):
    hidden1 = -0.46122 * wiek + 0.97314 * waga + (-0.39203 * wzrost) + 0.80109
    hidden1_po_aktywacji = sigmoid(hidden1)
    hidden2 = 0.78548 * wiek + 2.10584 * waga + (-0.57847 * wzrost) + 0.43529
    hidden2_po_aktywacji = sigmoid(hidden2)
    output = hidden1_po_aktywacji * -0.81546 + hidden2_po_aktywacji * 1.03775 - 0.2368
    return output

print(forwardPass(23, 75, 176))
print(forwardPass(20, 80, 182))
