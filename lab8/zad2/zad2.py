import pygad
import math
import numpy as np
import matplotlib.pyplot as plt

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)


def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

gene_space = [{'low': 0, 'high': 1},  # x
              {'low': 0, 'high': 1},  # y
              {'low': 0, 'high': 1},  # z
              {'low': 0, 'high': 1},  # u
              {'low': 0, 'high': 1},  # v
              {'low': 0, 'high': 1}]  # w

num_generations = 50
sol_per_pop = 10
num_genes = 6
num_parents_mating = 5
mutation_percent_genes = 17 # 100/6

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance.plot_fitness()
plt.savefig("metal_fitness.png")

# 1
# Parameters of the best solution : [0.29089477 0.35539955 0.8375641  0.96917193 0.6039173  0.        ]
# Fitness value of the best solution = 2.7161232372060327

# 2
# Parameters of the best solution : [0.22621937 0.22159454 0.98726818 0.96575407 0.         0.14424895]
# Fitness value of the best solution = 2.815407667197983