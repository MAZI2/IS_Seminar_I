import pygad
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import numpy as np
import time
import os

from basic_algorithm import run_algorithm as run_basic_algorithm
from basic_repair_algorithm import run_algorithm as run_basic_repair_algorithm
from general_repair_algorithm_1 import run_algorithm as run_general_repair_algorithm_1
from general_repair_algorithm_2 import run_algorithm as run_general_repair_algorithm_2
from repair_prob_constraint_algorithm import run_algorithm as run_repair_prob_constraint_algorithm

def benchmark_on_dataset(dataset_path, run_function, iterations, verbose = 1):
    seed = 0
    scores = np.zeros(iterations).astype(np.float64)
    runtimes = np.zeros(iterations).astype(np.float64) #in miliseconds

    for i in range(iterations):
        seed += 1
        timer_start = time.time()
        scores[i], dataset_id, num_papers, num_reviewers = run_function(dataset_path, seed)
        timer_end = time.time()

        runtimes[i] = np.round((timer_end - timer_start) * 1000, 3)

        if verbose == 1:
            print(f'benchmark progress {((i+1)/iterations)*100}% for {dataset_path}-{run_function.__name__}')

    return np.max(scores), np.mean(scores), np.var(scores), np.mean(runtimes), np.var(runtimes), dataset_id, num_papers, num_reviewers

def benchmark_all_datasets(folder_path, run_func, iterations):
    dataset_results = []
    for file_name in os.listdir(folder_path):
        dataset_file = os.path.join(folder_path, file_name)
        if os.path.isfile(dataset_file):
            
            max, mean, var, mean_runtime, var_runtime, dataset_id, num_papers, num_reviewers = benchmark_on_dataset(dataset_file, run_func, iterations)

            data = dict(
                id = dataset_id,
                papers = num_papers,
                reviewers = num_reviewers,
                max_score = max,
                mean_score = mean,
                score_var = var,
                mean_runtime = mean_runtime,
                runtime_var = var_runtime
            )
            dataset_results.append(data)
    return dataset_results


def make_benchmark_json(json_name, run_function, iterations=1, rerun = False):
    if rerun == False:
        return
    folder_path = "datasets"
    with open(json_name, "w") as file:
        data = benchmark_all_datasets(folder_path, run_function, iterations)
        json.dump(data, file, indent=4)
        print(json_name, "created")


#make_benchmark_json('benchmarks/basic_algorithm_benchmarks.json', run_basic_algorithm, iterations=it, rerun=rerun)
#make_benchmark_json('benchmarks/basic_repair_algorithm_benchmarks.json', run_basic_repair_algorithm, iterations=it, rerun=rerun)
#make_benchmark_json('benchmarks/general_repair_algorithm_1_benchmarks.json', run_general_repair_algorithm_1, iterations=it, rerun=rerun)
#make_benchmark_json('benchmarks/general_repair_algorithm_2_benchmarks.json', run_general_repair_algorithm_2, iterations=it, rerun=rerun)
#make_benchmark_json('benchmarks/repair_prob_constraint_algorithm_benchmarks.json', run_repair_prob_constraint_algorithm, iterations=it, rerun=rerun)

from concurrent.futures import ProcessPoolExecutor
it = 30
rerun = True

def unpack_args(args):
    return make_benchmark_json(*args)

if __name__ == "__main__":
    params = [
        ('benchmarks/basic_algorithm_benchmarks.json', run_basic_algorithm, it, rerun),
        ('benchmarks/basic_repair_algorithm_benchmarks.json', run_basic_repair_algorithm, it, rerun),
        ('benchmarks/general_repair_algorithm_1_benchmarks.json', run_general_repair_algorithm_1, it, rerun),
        ('benchmarks/general_repair_algorithm_2_benchmarks.json', run_general_repair_algorithm_2, it, rerun),
        ('benchmarks/repair_prob_constraint_algorithm_benchmarks.json', run_repair_prob_constraint_algorithm, it, rerun)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(unpack_args, params))
