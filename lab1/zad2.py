import numpy as np
import matplotlib.pyplot as plt
import random
import math

v = 50  
h = 100  
g = 9.81  

target_distance = random.randint(50, 340)
print(f"Cel znajduje się w odległości: {target_distance} metrów")

def calculate_distance(angle):
    angle_rad = math.radians(angle)  # Konwersja kąta na radiany
    distance = (v * math.sin(angle_rad) + math.sqrt(v**2 * math.sin(angle_rad)**2 + 2 * g * h)) * v * math.cos(angle_rad) / g
    return distance

def plot_trajectory(angle, shot_distance):
    angle_rad = math.radians(angle)
    vx = v * math.cos(angle_rad)
    vy = v * math.sin(angle_rad)
    t_flight = shot_distance / vx  

    t = np.linspace(0, t_flight, num=100)
    x = vx * t
    y = vy * t - 0.5 * g * t**2 + h
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Trajektoria pocisku")
    plt.title("Trajektoria pocisku Warwolf")
    plt.xlabel("Odległość")
    plt.ylabel("Wysokość")
    plt.grid(True)
    plt.legend()
    plt.savefig('trajektoria.png')
    plt.show()
attempts = 0
while True:
    angle = float(input("Podaj kąt strzału (w stopniach): "))
    attempts += 1
    shot_distance = calculate_distance(angle)
    
    if target_distance - 5 <= shot_distance <= target_distance + 5:
        print("Cel trafiony!")
        print(f"Liczba prób: {attempts}")
        plot_trajectory(angle, shot_distance)
        break
    else:
        print(f"Pudło. Pocisk poleciał na odległość: {shot_distance:.2f} metrów. Spróbuj ponownie.")


# import math
# import matplotlib.pyplot as plt
# import numpy as np

# # Stałe
# v = 50  
# h = 100  
# g = 9.81  

# def calculate_distance(angle):
#     angle_rad = math.radians(angle)
#     distance = (v * math.sin(angle_rad) + math.sqrt(v**2 * math.sin(angle_rad)**2 + 2 * g * h)) * v * math.cos(angle_rad) / g
#     return distance

# def plot_trajectory(angle, shot_distance):
#     angle_rad = math.radians(angle)
#     vx = v * math.cos(angle_rad)
#     vy = v * math.sin(angle_rad)
#     t_flight = shot_distance / vx  

#     t = np.linspace(0, t_flight, num=100)
#     x = vx * t
#     y = vy * t - 0.5 * g * t**2 + h
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(x, y, label="Trajektoria pocisku")
#     plt.title("Trajektoria pocisku Warwolf")
#     plt.xlabel("Odległość")
#     plt.ylabel("Wysokość")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig('trajektoria.png')
#     plt.show()

# def main():
#     target_distance = random.randint(50, 340)
#     print(f"Cel znajduje się w odległości: {target_distance} metrów")

#     attempts = 0
#     while True:
#         angle = float(input("Podaj kąt strzału (w stopniach): "))
#         attempts += 1
#         shot_distance = calculate_distance(angle)
        
#         if target_distance - 5 <= shot_distance <= target_distance + 5:
#             print("Cel trafiony!")
#             print(f"Liczba prób: {attempts}")
#             plot_trajectory(angle, shot_distance)
#             break
#         else:
#             print(f"Pudło. Pocisk poleciał na odległość: {shot_distance:.2f} metrów. Spróbuj ponownie.")

# if __name__ == "__main__":
#     main()