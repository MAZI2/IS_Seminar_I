import numpy as np

def eval_ff(solution, preferences, num_reviewers, num_papers, max_reviews_per_paper, min_reviews_per_paper, reviewer_capacity, authorship, friendship):
    # Shows smoe fitness info of the best solution
    rlpp = 10
    plpr = 10
    fc = 20
    ac = 50

    assignment = solution.reshape(num_reviewers, num_papers)

    #print(assignment)

    satisfaction = np.sum(assignment * preferences)
    penalties = 0

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

    #print("Satisfaction: ", satisfaction)
    return satisfaction - penalties
    