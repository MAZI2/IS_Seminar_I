import pygad
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import pyplot as plt
from fitness import *

# a chance to repair a constraint
repair_chance = 0.2
# a weight of additional points for new constraint
spread_weight = 0.8

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
    global spread_weight
    assignment = solution.reshape(num_reviewers, num_papers)
    
    # Penalty weights
    rlpp = 10
    plpr = 10
    fc = 20
    ac = 50

    # Satisfaction score (primary objective)
    satisfaction = np.sum(assignment * preferences)
    
    # Secondary objective: maximize the spread of preferences across reviewers
    spread_score = 0
    for i in range(num_reviewers):
        # Get the preferences for papers assigned to reviewer i
        assigned_papers = np.where(assignment[i] == 1)[0]
        if len(assigned_papers) > 0:
            reviewer_preferences = preferences[i, assigned_papers]
            # Calculate the variance of preferences
            mean_preference = np.mean(reviewer_preferences)
            variance = np.var(reviewer_preferences)
            spread_score += variance  # Add variance to the spread score
    
    # Constraints penalties (primary objective constraints)
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
    
    # The final fitness is the primary objective (satisfaction) plus the secondary objective (spread_score)
    # We want to maximize both objectives, so we add the spread score to the satisfaction
    # The spread score can also be scaled if necessary, e.g., by a constant factor
    return satisfaction + spread_weight * spread_score - penalties

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
    for i in range(num_reviewers):
        reviewer_count = np.sum(assignment[i])
        if reviewer_count > reviewer_capacity:
            for j in range(num_papers):
                if assignment[i, j] == 1:
                    return False
    
    # 2. Paper review requirements (each paper needs between min_reviews_per_paper and max_reviews_per_paper reviews)
    for j in range(num_papers):
        paper_count = np.sum(assignment[:, j])
        
        # Ensure the paper has at least min_reviews_per_paper reviews
        if paper_count < min_reviews_per_paper:
            for i in range(num_reviewers):
                if assignment[i, j] == 0:
                    return False
        
        # Remove excess reviewers from the paper if it exceeds max_reviews_per_paper
        if paper_count > max_reviews_per_paper:
            for i in range(num_reviewers):
                if assignment[i, j] == 1 and paper_count > max_reviews_per_paper:
                    return False
    
    # 3. Friendship constraints (no two friends should review the same paper)
    for i in range(num_reviewers):
        for j in range(i + 1, num_reviewers):
            if friendship[i, j] == 1:
                for k in range(num_papers):
                    if assignment[i, k] == 1 and assignment[j, k] == 1:
                        return False
    
    # 4. Authorship constraints (reviewers cannot review their own papers)
    for i in range(num_reviewers):
        for j in range(num_papers):
            if authorship[i, j] == 1:
                return False

    return True

# Repair function to correct an invalid solution
def repair_solution(assignment):
    global repair_chance

    # Ensure the offspring meet the constraints
    # 1. Reviewer capacity (max reviews per reviewer)
    for i in range(num_reviewers):
        reviewer_count = np.sum(assignment[i])
        if reviewer_count > reviewer_capacity:
            for j in range(num_papers):
                if assignment[i, j] == 1 and reviewer_count > reviewer_capacity and np.random.rand() < repair_chance:
                    assignment[i, j] = 0
                    reviewer_count -= 1
    
    # 2. Paper review requirements (each paper needs between min_reviews_per_paper and max_reviews_per_paper reviews)
    for j in range(num_papers):
        paper_count = np.sum(assignment[:, j])
        
        # Ensure the paper has at least min_reviews_per_paper reviews
        if paper_count < min_reviews_per_paper:
            for i in range(num_reviewers):
                if assignment[i, j] == 0 and np.random.rand() < repair_chance:
                    assignment[i, j] = 1
                    paper_count += 1
                    if paper_count >= min_reviews_per_paper:
                        break
        
        # Remove excess reviewers from the paper if it exceeds max_reviews_per_paper
        if paper_count > max_reviews_per_paper:
            for i in range(num_reviewers):
                if assignment[i, j] == 1 and paper_count > max_reviews_per_paper and np.random.rand() < repair_chance:
                    assignment[i, j] = 0
                    paper_count -= 1
    
    # 3. Friendship constraints (no two friends should review the same paper)
    for i in range(num_reviewers):
        for j in range(i + 1, num_reviewers):
            if friendship[i, j] == 1:
                for k in range(num_papers):
                    if assignment[i, k] == 1 and assignment[j, k] == 1:
                        # Swap one reviewer's assignment to a different paper
                        for l in range(num_papers):
                            if assignment[i, l] == 0 and np.random.rand() < repair_chance:
                                assignment[i, l] = 1
                                assignment[i, k] = 0
                                break
    
    # 4. Authorship constraints (reviewers cannot review their own papers)
    for i in range(num_reviewers):
        for j in range(num_papers):
            if authorship[i, j] == 1 and np.random.rand() < repair_chance:
                assignment[i, j] = 0

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

# solution = run_algorithm("../datasets/medium_dataset_2.json")
# assignment = solution.reshape(num_reviewers, num_papers)
# print(eval_ff(solution, preferences, num_reviewers, num_papers, max_reviews_per_paper, min_reviews_per_paper, reviewer_capacity, authorship, friendship))