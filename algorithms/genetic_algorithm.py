import numpy as np


def genetic_algorithm(
    objective_func,
    bounds,
    pop_size=50,
    num_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    mutation_scale=0.1,
    tournament_size=3,
    elitism=True,
):
    """
    Standard Real-Coded Genetic Algorithm (GA) for global minimization.

    Args:
        objective_func (callable): The function to minimize.
        bounds (list of tuples): List of (min, max) for each dimension.
        pop_size (int): Number of individuals in the population.
        num_generations (int): Maximum number of generations.
        crossover_rate (float): Probability of crossover occurring between parents.
        mutation_rate (float): Probability of a gene being mutated.
        mutation_scale (float): Standard deviation of Gaussian noise added during mutation.
        tournament_size (int): Number of individuals in tournament selection.
        elitism (bool): If True, preserves the best individual to the next generation.

    Returns:
        best_solution (np.array): The best solution found.
        best_fitness (float): The fitness value of the best solution.
        history (list): List of best fitness values per generation.
    """
    # 1. Initialization
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)

    # Initialize population randomly
    population = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)

    # Evaluate initial fitness
    fitness = np.array([objective_func(ind) for ind in population])

    # Track best
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    history = [best_fitness]

    # 2. Main Loop
    for gen in range(num_generations):
        new_population = []

        # Elitism: Keep the single best individual?
        if elitism:
            new_population.append(best_solution.copy())
            start_idx = 1
        else:
            start_idx = 0

        # Generate the rest of the new population
        while len(new_population) < pop_size:
            # --- Selection (Tournament) ---
            # Pick 'k' random individuals, select the one with best fitness
            # Parent 1
            candidates_1 = np.random.randint(0, pop_size, size=tournament_size)
            p1_idx = candidates_1[np.argmin(fitness[candidates_1])]
            parent1 = population[p1_idx]

            # Parent 2
            candidates_2 = np.random.randint(0, pop_size, size=tournament_size)
            p2_idx = candidates_2[np.argmin(fitness[candidates_2])]
            parent2 = population[p2_idx]

            # --- Crossover (Single Point) ---
            if np.random.rand() < crossover_rate:
                # Pick a crossover point
                pt = np.random.randint(1, dim) if dim > 1 else 0
                # Create children (we only keep one per pair to simplify loop)
                child = np.concatenate((parent1[:pt], parent2[pt:]))
            else:
                child = parent1.copy()

            # --- Mutation (Gaussian) ---
            # Iterate over each gene (dimension)
            for i in range(dim):
                if np.random.rand() < mutation_rate:
                    # Add random noise
                    child[i] += np.random.normal(0, mutation_scale)
                    # Clip to bounds
                    child[i] = np.clip(child[i], min_b[i], max_b[i])

            new_population.append(child)

        # Update Population
        population = np.array(new_population)

        # Evaluate New Fitness
        fitness = np.array([objective_func(ind) for ind in population])

        # Update Best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        history.append(best_fitness)

    return best_solution, best_fitness, history
