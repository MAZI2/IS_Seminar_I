import pygad
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import pyplot as plt
from fitness import *

dataset_id = 0
num_papers = 0
num_reviewers = 0
reviewer_capacity = 0
min_reviews_per_paper = 0
max_reviews_per_paper = 0
preferences = np.array([])
friendship = np.array([])
authorship = np.array([])

def load_dataset(path):
    global dataset_id, num_papers, num_reviewers, reviewer_capacity, min_reviews_per_paper, max_reviews_per_paper, preferences, friendship, authorship
    with open(path, "r") as file:
        data = json.load(file)

    # Extract information
    dataset_id = data["dataset_id"]
    num_papers = data["num_papers"]
    num_reviewers = data["num_reviewers"]
    reviewer_capacity = data["reviewer_capacity"]
    min_reviews_per_paper = data["min_reviews_per_paper"]
    max_reviews_per_paper = data["max_reviews_per_paper"]
    preferences = np.array(data["preferences"])
    friendship = np.array(data["friendships"])
    authorship = np.array(data["authorship"])

# Fitness function
def fitness_func(ga_instance, solution, solution_idx):
    assignment = solution.reshape(num_reviewers, num_papers)
    
    # Penalty weights
    rlpp = 10

    # Satisfaction score
    satisfaction = np.sum(assignment * preferences)
    
    # Constraints penalties
    penalties = 0
    
    # Review limit per paper
    for j in range(num_papers):
        if np.sum(assignment[:, j]) < min_reviews_per_paper or np.sum(assignment[:, j]) > max_reviews_per_paper:
            penalties += rlpp
    
    return satisfaction - penalties

def check(ga_instance, children):
    # Repair children
    for idx, child in enumerate(children):
        matr = np.reshape(child, (num_reviewers, num_papers))
        if not is_valid_assignment(matr):
            children[idx] = repair_solution(matr).reshape(-1)
    return children

def check_initial(ga_instance):
    # Repair initial population
    for solution_idx, solution in enumerate(ga_instance.population):
        assignment = solution.reshape((num_reviewers, num_papers))
        if not is_valid_assignment(assignment):
            ga_instance.population[solution_idx] = repair_solution(assignment).reshape(-1)

# Validity checks (constraints)
def is_valid_assignment(assignment):
    # Check paper reviews (min and max reviews per paper)  - Skipped because with our correction algorithm, we are not fixing this constraint 
    #
    # paper_counts = np.sum(assignment, axis=0)
    # if any(paper_counts < min_reviews_per_paper) or any(paper_counts > max_reviews_per_paper):
    #     return False
    
    # Check reviewer capacities
    reviewer_counts = np.sum(assignment, axis=1)
    if any(reviewer_counts > reviewer_capacity):
        return False
    
    # Check authorship constraints (a reviewer cannot review their own paper)
    if np.any(assignment & authorship):
        return False

    # Check friendship constraints (no two friends should review the same paper)
    for i in range(num_reviewers):
        for j in range(num_reviewers):
            if friendship[i][j] == 1:
                for k in range(num_papers):
                    if assignment[i][k] == 1 and assignment[j][k] == 1:
                        return False

    return True

# Repair function to correct an invalid solution
def repair_solution(assignment):
    # Ensure reviewer capacities are respected
    reviewer_counts = np.sum(assignment, axis=1)
    for i in range(num_reviewers):
        if reviewer_counts[i] > reviewer_capacity:
            for j in range(num_papers):
                if assignment[i][j] == 1:
                    assignment[i][j] = 0
                    reviewer_counts = np.sum(assignment, axis=1)
                    if reviewer_counts[i] <= reviewer_capacity:
                        break
    
    # Ensure friendship constraints are respected
    for i in range(num_reviewers):
        for j in range(num_reviewers):
            if friendship[i][j] == 1:
                for k in range(num_papers):
                    if assignment[i][k] == 1 and assignment[j][k] == 1:
                        assignment[i][k] = 0  # Remove one of them
                        break
    
    # Ensure authorship constraints are respected
    assignment *= (1-authorship)
    
    return assignment

def run_algorithm(dataset_path, seed):
    # Genetic Algorithm setup
    binary_gene_space = [0,1]

    load_dataset(dataset_path)

    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=1000,
        num_genes=num_reviewers * num_papers,
        gene_type=int,
        gene_space=binary_gene_space,
        mutation_type="swap", 
        crossover_type="uniform", 
        mutation_probability=0.2,
        on_start=check_initial,
        on_mutation=check,
        on_crossover=check,
        random_seed=seed
    )

    # Run the GA
    ga_instance.run()

    # Best solution
    best_solution, solution_fitness, _ = ga_instance.best_solution()
    e = eval_ff(best_solution, preferences, num_reviewers, num_papers, max_reviews_per_paper, min_reviews_per_paper, reviewer_capacity, authorship, friendship)
    return e, dataset_id, num_papers, num_reviewers
