import matplotlib.pyplot as plt
import random
import time

from aco import AntColony

start_time = time.time()
plt.style.use("dark_background")


# COORDS = (
#     (20, 52),
#     (43, 50),
#     (20, 84),
#     (70, 65),
#     (29, 90),
#     (87, 83),
#     (73, 23),
# )

COORDS = (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90), (87, 83), (73, 23),
    (15, 15), (95, 70), (80, 30), (10, 40), (55, 60), (45, 70), (25, 75)
)

# gubi sie przy takich wspolrzednych

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)
# 32.47

# Eksperyment 2: Zwiększona liczba mrówek
colony = AntColony(COORDS, ant_count=500, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                    iterations=300)
#53.92

# # Eksperyment 3: Zmniejszona liczba mrówek
# colony = AntColony(COORDS, ant_count=100, alpha=0.5, beta=1.2, 
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)
# 11.03

# # Eksperyment 4: Zmienione parametry alpha i beta
# colony = AntColony(COORDS, ant_count=300, alpha=1.0, beta=2.0, 
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)
# 32.33

# # Eksperyment 5: Zmieniona szybkość parowania feromonów
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
#                     pheromone_evaporation_rate=0.60, pheromone_constant=1000.0,
#                     iterations=300)
# 32.67

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

end_time = time.time()
print(f"--- {end_time - start_time} seconds ---")

plt.show()

# czas jest zalezny tak naprawde tylko od zmiany liczby mrowek no i ew iteracji