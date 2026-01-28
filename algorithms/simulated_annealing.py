import numpy as np
import math


def simulated_annealing(
    objective_func,
    bounds,
    max_iter=1000,
    initial_temp=100.0,
    cooling_rate=0.99,
    step_size=0.1,
    tol=1e-6,
):
    """
    Standard Simulated Annealing (SA) for global minimization.

    Args:
        objective_func (callable): The function to minimize.
        bounds (list of tuples): List of (min, max) for each dimension.
        max_iter (int): Maximum number of iterations.
        initial_temp (float): Starting temperature. Higher means more exploration (accepts worse moves).
        cooling_rate (float): Rate at which temp decays (0 < alpha < 1). E.g., 0.99.
        step_size (float): Scale of the random noise added to generate neighbors.
        tol (float): Tolerance for convergence (stop if temp is very low).

    Returns:
        best_solution (np.array): The best solution found globally.
        best_energy (float): The value of the objective function at the best solution.
        history (list): List of best energy values over time.
    """
    # 1. Initialization
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)

    # Start at a random position
    current_solution = min_b + (max_b - min_b) * np.random.rand(dim)
    current_energy = objective_func(current_solution)

    # Keep track of the absolute best solution found so far
    best_solution = current_solution.copy()
    best_energy = current_energy

    current_temp = initial_temp
    history = [best_energy]

    # 2. Main Loop
    for i in range(max_iter):
        # --- Generate Neighbor ---
        # Add random Gaussian noise to current solution
        perturbation = np.random.normal(0, step_size, size=dim)
        neighbor = current_solution + perturbation

        # Boundary Handling: Clip to stay valid
        neighbor = np.clip(neighbor, min_b, max_b)

        # --- Calculate Energy Change ---
        neighbor_energy = objective_func(neighbor)
        delta_energy = neighbor_energy - current_energy

        # --- Acceptance Probability (Metropolis Criterion) ---
        if delta_energy < 0:
            # Improvement: Always accept
            accept = True
        else:
            # Worsening: Accept with probability e^(-delta / T)
            # Avoid overflow if T is near 0
            if current_temp < 1e-10:
                probability = 0.0
            else:
                probability = math.exp(-delta_energy / current_temp)

            accept = np.random.rand() < probability

        # --- Update State ---
        if accept:
            current_solution = neighbor
            current_energy = neighbor_energy

            # Update global best if this is the best ever seen
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current_solution.copy()

        # --- Cooling Schedule ---
        current_temp *= cooling_rate
        history.append(best_energy)

        # Convergence Check
        if current_temp < tol:
            break

    return best_solution, best_energy, history
