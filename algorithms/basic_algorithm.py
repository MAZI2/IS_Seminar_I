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
    # Penalty weights
    rlpp = 10
    plpr = 10
    fc = 20
    ac = 50

    assignment = solution.reshape(num_reviewers, num_papers)

    # Satisfaction score
    satisfaction = np.sum(assignment * preferences)

    # Constraints penalties
    penalties = 0

    # Review limit per paper
    for j in range(num_papers):
        if np.sum(assignment[:, j]) < min_reviews_per_paper or np.sum(assignment[:, j]) > max_reviews_per_paper:
            penalties += rlpp

    # Paper limit per reviewer
    for i in range(num_reviewers):
        if np.sum(assignment[i, :]) > reviewer_capacity:
            penalties += plpr

    # Friendship constraint
    for j in range(num_papers):
        reviewers = np.where(assignment[:, j] == 1)[0]
        for r1 in reviewers:
            for r2 in reviewers:
                if friendship[r1, r2] == 1:
                    penalties += fc

    # Authorship constraint
    if np.any(assignment * authorship):
        penalties += ac

    return satisfaction - penalties

def run_algorithm(dataset_path, seed):
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
        random_seed=seed
    )

    # Run the GA
    ga_instance.run()

    # Best solution
    best_solution, solution_fitness, _ = ga_instance.best_solution()
    e = eval_ff(best_solution, preferences, num_reviewers, num_papers, max_reviews_per_paper, min_reviews_per_paper, reviewer_capacity, authorship, friendship)
    return e, dataset_id, num_papers, num_reviewers