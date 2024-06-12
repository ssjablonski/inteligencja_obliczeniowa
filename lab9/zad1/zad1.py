import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# Funkcja wytrzymałości stopu metali
def endurance(position):
    x, y, z, u, v, w = position
    return -(np.exp(-2*(y-np.sin(x))**2) + np.sin(z*u) + np.cos(v*w))

# Funkcja obliczająca wartość funkcji celu dla całego roju
def f(x):
    n_particles = x.shape[0]
    j = [endurance(x[i]) for i in range(n_particles)]
    return np.array(j)

# Opcje algorytmu PSO
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Ograniczenia dla dziedziny poszukiwań
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

# Inicjalizacja optymalizatora PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# Uruchomienie optymalizacji
optimizer.optimize(f, iters=1000)

# Wyświetlenie wyników
print("Best cost:", optimizer.cost_history[-1])
print("Best position:", optimizer.pos_history[-1])

# Wykres historii kosztu
plot_cost_history(optimizer.cost_history)
plt.show()

# best cost: 0.9278339195133264, best pos: [0.10928355 0.97249464 0.48244188 0.02747904 0.83179651 0.97390312]
# Best cost: 0.9278339195133264
# Best position: [[0.86782831 0.31286329 0.48241478 0.02748043 0.8312737  0.62601235]
#  [0.14984838 0.66989816 0.67082194 0.0210639  0.872604   0.71435904]
#  [0.74212925 0.75168823 0.46311896 0.66912518 0.80186521 0.63006748]
#  [0.54593984 0.96987236 0.98364112 0.92588122 0.83045772 0.3525665 ]
#  [0.70249058 0.9705807  0.15486537 0.55339385 0.79182367 0.14266028]
#  [0.20345076 0.59363316 0.80143554 0.73627201 0.54892628 0.70628454]
#  [0.08408373 0.65866969 0.53951216 0.77774353 0.69269311 0.48580149]
#  [0.10928355 0.93436917 0.48244188 0.73437208 0.83179651 0.36123102]
#  [0.52891412 0.38249393 0.55039314 0.74207659 0.51026334 0.06103426]
#  [0.19262019 0.26108106 0.48906984 0.79275723 0.81604823 0.22769798]]


# #max
# Best cost: -2.8163459924306427
# Best position: [[0.34823057 0.44347104 0.31842206 0.76038776 0.41440816 0.39755554]
#  [0.4550218  0.43384264 0.75769147 0.27318243 0.85003062 0.31195076]
#  [0.46178528 0.43849624 0.78500949 0.70556716 0.43228672 0.3499432 ]
#  [0.46051973 0.44480336 0.34304041 0.87446937 0.46777876 0.64008248]
#  [0.4632703  0.44208914 0.6375169  0.4236367  0.474986   0.50656621]
#  [0.45981867 0.4369881  0.16221741 0.31584284 0.45473138 0.97810791]
#  [0.42575877 0.44191255 0.83835157 0.24973389 0.25616096 0.76794591]
#  [0.51170078 0.43677503 0.80319047 0.63356306 0.40527754 0.54656528]
#  [0.46304008 0.44342805 0.48563649 0.66160857 0.47505313 0.40225258]
#  [0.49519633 0.44284239 0.90729598 0.57654787 0.47661579 0.23929317]]