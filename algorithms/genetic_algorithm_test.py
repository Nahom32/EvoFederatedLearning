import numpy as np
from genetic_algorithm import genetic_algorithm


def sphere_function(x):
    """Simple convex function. Min at 0."""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Non-convex function. Min at (1, 1, ..., 1)."""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rastrigin_function(x):
    """Multimodal function. Min at 0."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# --- Runner ---


def run_tests():
    print("🧬 Running Independent Genetic Algorithm Tests...\n")

    # TEST 1: Sphere Function (5 Dimensions)
    print("--- Test 1: Sphere Function (5D) ---")
    bounds = [(-5.0, 5.0)] * 5
    best_sol, best_val, _ = genetic_algorithm(
        sphere_function,
        bounds,
        pop_size=50,
        num_generations=200,
        mutation_rate=0.2,
        mutation_scale=0.1,
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    assert best_val < 0.1, "Failed to optimize Sphere function!"
    print("✅ Passed\n")

    # TEST 2: Rosenbrock Function (2 Dimensions)
    # Harder valley. GA needs more generations or larger population.
    print("--- Test 2: Rosenbrock Function (2D) ---")
    bounds = [(-2.0, 2.0)] * 2
    best_sol, best_val, _ = genetic_algorithm(
        rosenbrock_function,
        bounds,
        pop_size=100,
        num_generations=500,
        mutation_rate=0.3,
        mutation_scale=0.2,  # Higher mutation helps escape local optima
    )
    print(f"Result: {best_val:.6f} at {best_sol} (Expected 0.0 at [1. 1.])")
    assert best_val < 0.5, "Failed to optimize Rosenbrock function!"
    print("✅ Passed\n")

    # TEST 3: Rastrigin Function (5 Dimensions)
    # Very hard. GA is good at this if diversity is maintained.
    print("--- Test 3: Rastrigin Function (5D) ---")
    bounds = [(-5.12, 5.12)] * 5
    best_sol, best_val, _ = genetic_algorithm(
        rastrigin_function,
        bounds,
        pop_size=200,
        num_generations=500,
        mutation_rate=0.1,
        mutation_scale=0.5,  # Larger steps needed to jump peaks
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    # GA might find a near-optimal solution (e.g., 0.99) instead of perfect 0.0
    assert best_val < 2.0, "Failed to optimize Rastrigin function!"
    print("✅ Passed\n")


if __name__ == "__main__":
    run_tests()
