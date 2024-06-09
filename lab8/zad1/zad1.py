import pygad
import time
import matplotlib.pyplot as plt

items = [
    {"name": "Zegar", "value": 100, "weight": 7},
    {"name": "Obraz-pejzaż", "value": 300, "weight": 7},
    {"name": "Obraz-portret", "value": 200, "weight": 6},
    {"name": "Radio", "value": 40, "weight": 2},
    {"name": "Laptop", "value": 500, "weight": 5},
    {"name": "Lampka nocna", "value": 70, "weight": 6},
    {"name": "Srebrne sztućce", "value": 100, "weight": 1},
    {"name": "Porcelana", "value": 250, "weight": 3},
    {"name": "Figura z brązu", "value": 300, "weight": 10},
    {"name": "Skórzana torebka", "value": 280, "weight": 3},
    {"name": "Odkurzacz", "value": 300, "weight": 15}
]

max_weight = 25
values = [item["value"] for item in items]
weights = [item["weight"] for item in items]
names = [item["name"] for item in items]

# Zmodyfikowana funkcja fitness
def fitness_func(ga_instance, solution, solution_idx):
    total_value = 0
    total_weight = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            total_value += values[i]
            total_weight += weights[i]
    if total_weight > max_weight or total_value > 1630:
        return 0
    return total_value

def on_generation(ga_instance):
    best_solution = ga_instance.best_solution()
    best_solution_fitness = best_solution[1]
    if best_solution_fitness >= 1630:
        ga_instance.keep_solving = False


ga_instance = pygad.GA(
    fitness_func=fitness_func,
    gene_type=int,
    gene_space=[0, 1],
    num_generations=50,
    num_parents_mating=2,
    sol_per_pop=10,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    num_genes=len(items),
    on_generation=on_generation
)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Solution: {solution}")
print(f"Fitness of the best solution: {solution_fitness}")
print(f"Best solution index: {solution_idx}")

selected_items = [names[i] for i in range(len(solution)) if solution[i] == 1]
total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)
print(f"Selected items: {selected_items}")

# Przebieg optymalizacji
ga_instance.plot_fitness()
plt.show()

# Testowanie skuteczności algorytmu
success_count = 0
total_time = 0

for i in range(10):
    start_time = time.time()

    ga_instance = pygad.GA(
        fitness_func=fitness_func,
        gene_type=int,
        gene_space=[0, 1],
        num_generations=50,
        num_parents_mating=2,
        sol_per_pop=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        num_genes=len(items),
        on_generation=on_generation
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    selected_items = [item['name'] for item, selected in zip(items, solution) if selected]
    total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
    total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)

    print(f"Final-Solution: {selected_items}")
    print(f"Total-Value: {total_value}")
    print(f"Total-Weight: {total_weight}")
    total_time = time.time() - start_time

    if solution_fitness >= 1630:
        success_count += 1
        total_time += total_time

    ga_instance.plot_fitness().savefig(f"plot{i}.png")

print(f"Percentage of best solutions found: {success_count / 10 * 100}%")
if success_count > 0:
    print(f"Average time of successful solution: {total_time / success_count} seconds")
else:
    print("No successful solutions found.")

# Percentage of best solutions found: 90.0%
# Average time of successful solution: 0.002826319800482856 seconds