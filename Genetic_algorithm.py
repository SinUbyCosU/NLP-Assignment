
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# GENETIC ALGORITHM TUTORIAL (CLEAN VERSION)
# =====================================================

# -----------------------------------------------------
# Problem Definition
# Maximize: y = sum(w_i * x_i)
#Task is to find best w_i
# -----------------------------------------------------
equation_inputs = np.array([4, -2, 3.5, 5, -11, -4.7])
num_weights = len(equation_inputs)

# -----------------------------------------------------
# GA Parameters
# -----------------------------------------------------
population_size = 60
num_parents = 2
num_generations = 25
mutation_rate = 0.3


# -----------------------------------------------------
# Initialize Population
# Each row is a chromosome (solution)
# -----------------------------------------------------
def initialize_population(current_population_size):
    return np.random.uniform(-4, 4, (current_population_size, num_weights))


# -----------------------------------------------------
# Fitness Function
# Computes dot product: w.x
# -----------------------------------------------------
def fitness(population, objective="default"):
    if objective == "target":
        return -np.sum((population - 2) ** 2, axis=1)

    if objective == "multi":
        f1 = np.sum(population * equation_inputs, axis=1)
        f2 = -np.sum(population ** 2, axis=1)
        return f1 + 0.1 * f2

    return np.sum(population * equation_inputs, axis=1)


# -----------------------------------------------------
# Selection: choose best individuals
# -----------------------------------------------------
def select_parents(population, fitness_values, current_num_parents):
    parents = np.empty((current_num_parents, num_weights))
    fitness_copy = fitness_values.copy()

    for i in range(current_num_parents):
        idx = np.argmax(fitness_copy)
        parents[i] = population[idx]
        fitness_copy[idx] = -np.inf  # avoid selecting again

    return parents


def tournament_select_parents(population, fitness_values, current_num_parents, k=3):
    parents = []

    for _ in range(current_num_parents):
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmax(fitness_values[indices])]
        parents.append(population[best_idx])

    return np.array(parents)


# -----------------------------------------------------
# Crossover: combine two parents
# -----------------------------------------------------
def crossover(parents, current_population_size):
    offspring = []
    crossover_point = num_weights // 2

    for i in range(current_population_size - len(parents)):
        p1 = parents[i % len(parents)]
        p2 = parents[(i + 1) % len(parents)]

        child = np.concatenate([p1[:crossover_point],
                                p2[crossover_point:]])
        offspring.append(child)

    return np.array(offspring)


# -----------------------------------------------------
# Mutation: randomly modify one gene
# -----------------------------------------------------
def mutation(offspring, current_mutation_rate):
    for i in range(len(offspring)):
        if np.random.rand() < current_mutation_rate:
            gene_idx = np.random.randint(0, num_weights)
            offspring[i][gene_idx] += np.random.uniform(-1, 1)
    return offspring


# -----------------------------------------------------
# Constraint Handling (Exercise 5)
# -----------------------------------------------------
def apply_constraints(population, fitness_values):
    for i in range(len(population)):
        if np.any(np.abs(population[i]) > 10):
            fitness_values[i] -= 100

    return fitness_values


def run_ga(
    title,
    current_population_size=population_size,
    current_num_parents=num_parents,
    current_mutation_rate=mutation_rate,
    objective="default",
    use_tournament=False,
    use_constraints=True,
    use_early_stopping=False,
):
    fitness_history = []
    population = initialize_population(current_population_size)

    print(f"\n=== {title} ===")

    for generation in range(num_generations):
        fit = fitness(population, objective=objective)

        if use_constraints:
            fit = apply_constraints(population, fit)

        best_fit = np.max(fit)
        fitness_history.append(best_fit)
        print("Generation", generation, "| Best Fitness:", round(best_fit, 4))

        if use_tournament:
            parents = tournament_select_parents(
                population, fit, current_num_parents
            )
        else:
            parents = select_parents(population, fit, current_num_parents)

        offspring = crossover(parents, current_population_size)
        offspring = mutation(offspring, current_mutation_rate)

        population[:len(parents)] = parents
        population[len(parents):] = offspring

        if use_early_stopping and generation > 5:
            if abs(fitness_history[-1] - fitness_history[-5]) < 1e-3:
                print("Early stopping triggered")
                break

    fit = fitness(population, objective=objective)
    if use_constraints:
        fit = apply_constraints(population, fit)
    best_idx = np.argmax(fit)

    print("Best Solution:", population[best_idx])
    print("Best Fitness:", fit[best_idx])

    return fitness_history


# -----------------------------------------------------
# Run all exercises and collect histories
# -----------------------------------------------------
scenarios = [
    ("Base GA", {}),
    ("Exercise 1A - Mutation 0.0", {"current_mutation_rate": 0.0}),
    ("Exercise 1B - Mutation 0.9", {"current_mutation_rate": 0.9}),
    ("Exercise 1C - Mutation 0.3", {"current_mutation_rate": 0.3}),
    ("Exercise 2A - Population 5", {"current_population_size": 5}),
    ("Exercise 2B - Population 50", {"current_population_size": 50}),
    ("Exercise 2C - Population 200", {"current_population_size": 200}),
    ("Exercise 3A - Parents 1", {"current_num_parents": 1}),
    ("Exercise 3B - Parents 2", {"current_num_parents": 2}),
    ("Exercise 3C - Parents 5", {"current_num_parents": 5}),
    ("Exercise 4 - Tournament", {"use_tournament": True}),
    ("Exercise 5 - Early Stopping", {"use_early_stopping": True}),
    ("Exercise 6 - Target Objective", {"objective": "target"}),
    ("Exercise 7 - Multi Objective", {"objective": "multi"}),
]

all_histories = []
for scenario_title, params in scenarios:
    history = run_ga(scenario_title, **params)
    all_histories.append((scenario_title, history))


# -----------------------------------------------------
# Plot all histories in a grid
# -----------------------------------------------------
num_plots = len(all_histories)
num_cols = 4
num_rows = int(np.ceil(num_plots / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))
axes = np.array(axes).reshape(-1)

for idx, (title, history) in enumerate(all_histories):
    axes[idx].plot(history, linewidth=1.8)
    axes[idx].set_title(title, fontsize=10)
    axes[idx].set_xlabel("Generation")
    axes[idx].set_ylabel("Best Fitness")
    axes[idx].grid(alpha=0.25)

for idx in range(num_plots, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.show()



