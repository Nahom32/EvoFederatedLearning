# EvoFederatedLearning

## Abstract

Federated Learning (FL) is a distributed machine learning paradigm that enables model training on decentralized data without compromising data privacy. However, FL faces challenges such as communication overhead, statistical heterogeneity, and client drift. To address these challenges, we propose EvoFederatedLearning, a novel approach that leverages evolutionary algorithms to optimize the client selection process in FL. We implement and evaluate four different evolutionary algorithms: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE), and Simulated Annealing (SA). Our experiments on the MNIST dataset show that EvoFederatedLearning significantly improves the convergence speed and final accuracy of the global model compared to the standard FedAvg algorithm.

## Repository Structure

The repository is organized as follows:

- `algorithms/`: Contains the Python implementations of the four evolutionary algorithms:
  - `genetic_algorithm.py`: A standard real-coded genetic algorithm.
  - `particle_swarm_optimization.py`: A standard particle swarm optimization algorithm.
  - `differential_evolution.py`: A standard differential evolution algorithm.
  - `simulated_annealing.py`: A standard simulated annealing algorithm.
- `notebooks/`: Contains the Jupyter notebook with the experiments and results.
  - `fedProxMetaheuristics.ipynb`: The main notebook for running the experiments and generating the results.
- `requirements.txt`: Contains the list of required Python packages.
- `README.md`: This file.

## Getting Started

To run the experiments, you need to have Python 3 and the required packages installed.

### 1. Clone the repository

```bash
git clone https://github.com/Nahom32/EvoFederatedLearning.git
cd EvoFederatedLearning
```

### 2. Create a virtual environment and install dependencies

It is recommended to use a virtual environment to manage the dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the experiment

You can open the Jupyter notebook `notebooks/fedProxMetaheuristics.ipynb` and run the cells to reproduce the results.

```bash
jupyter notebook notebooks/fedProxMetaheuristics.ipynb
```

## The Algorithms

The following evolutionary algorithms are implemented in the `algorithms/` directory:

- **Genetic Algorithm (GA):** A search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.
- **Particle Swarm Optimization (PSO):** A computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It optimizes a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity.
- **Differential Evolution (DE):** A method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. DE is used for multidimensional real-valued functions but does not use the gradient of the function being optimized, which means DE does not require for the optimization problem to be differentiable as is required by classic optimization methods such as gradient descent and quasi-newton methods.
- **Simulated Annealing (SA):** A probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. It is often used when the search space is discrete (e.g., all tours that visit a given set of cities).

## The Experiment

The experiment is conducted in the `notebooks/fedProxMetaheuristics.ipynb` Jupyter notebook. The experiment uses the MNIST dataset and the FedProx algorithm. It compares the performance of random client selection (baseline) against the four evolutionary algorithms. The notebook is divided into the following sections:

1. **Setup:** Initializes the environment and sets the parameters for the experiment.
2. **Data Loading and Preparation:** Loads the MNIST dataset and prepares it for the federated learning setting.
3. **Model Definition:** Defines the simple neural network model used in the experiment.
4. **Federated Learning Functions:** Contains the functions for training and evaluating the clients and the global model.
5. **Evolutionary Algorithms:** Contains the implementations of the four evolutionary algorithms.
6. **Experiment Loop:** Runs the main experiment loop, which iterates through the communication rounds and applies the different client selection strategies.
7. **Results:** Displays the results in a table and a plot.

## Results

The results of the experiment show that the evolutionary algorithms significantly outperform the baseline random selection strategy. The following plot shows the convergence of the global model accuracy for each of the client selection strategies:

![Convergence Plot](https://raw.githubusercontent.com/Nahom32/EvoFederatedLearning/main/convergence_plot.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

