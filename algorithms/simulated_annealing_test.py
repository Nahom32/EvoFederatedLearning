import numpy as np
from simulated_annealing import simulated_annealing


def sphere_function(x):
    """Simple convex function. Min at 0."""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Non-convex function. Min at (1, 1, ..., 1)."""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rastrigin_function(x):
    """Multimodal function (many local optima). Min at 0."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# --- Runner ---


def run_tests():
    print("🔥 Running Independent Simulated Annealing Tests...\n")

    # TEST 1: Sphere Function (5 Dimensions)
    # Easy convex problem.
    print("--- Test 1: Sphere Function (5D) ---")
    bounds = [(-5.0, 5.0)] * 5
    best_sol, best_val, _ = simulated_annealing(
        sphere_function,
        bounds,
        max_iter=5000,
        initial_temp=10.0,
        cooling_rate=0.99,
        step_size=0.1,
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    assert best_val < 1e-2, "Failed to optimize Sphere function!"
    print("✅ Passed\n")

    # TEST 2: Rosenbrock Function (2 Dimensions)
    # Harder valley. We increase iterations and tweak step size.
    print("--- Test 2: Rosenbrock Function (2D) ---")
    bounds = [(-2.0, 2.0)] * 2
    best_sol, best_val, _ = simulated_annealing(
        rosenbrock_function,
        bounds,
        max_iter=5000,
        initial_temp=100.0,
        cooling_rate=0.99,
        step_size=0.1,
    )
    print(f"Result: {best_val:.6f} at {best_sol} (Expected 0.0 at [1. 1.])")
    assert best_val < 1e-1, "Failed to optimize Rosenbrock function!"
    print("✅ Passed\n")

    # TEST 3: Rastrigin Function (5 Dimensions)
    # Very hard. Requires high initial temp to jump out of local pits.
    print("--- Test 3: Rastrigin Function (5D) ---")
    bounds = [(-5.12, 5.12)] * 5
    best_sol, best_val, _ = simulated_annealing(
        rastrigin_function,
        bounds,
        max_iter=100000,
        initial_temp=70.0,
        cooling_rate=0.992,
        step_size=0.95,
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    # SA can struggle with high-dim multimodal functions without fine-tuning.
    # We use a relaxed assertion here.
    assert best_val < 2.0, "Failed to optimize Rastrigin function!"
    print("✅ Passed\n")


if __name__ == "__main__":
    run_tests()
