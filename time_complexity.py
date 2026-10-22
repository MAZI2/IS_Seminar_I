import pygad
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import numpy as np
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
import json
from matplotlib import pyplot as plt
from scipy.stats import linregress

def read_benchmark_json(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r") as file:
        loaded_data = json.load(file)
    return loaded_data

def show_benchmark_time_complexities(path):
    data = read_benchmark_json(path)

    mean_runtimes_rev_cnt = np.zeros(len(data)).astype(np.float64)
    reviewer_count = np.zeros(len(data)).astype(np.uint32)
    for i, benchmark in enumerate(sorted(data, key=lambda x: x["reviewers"])):
        mean_runtimes_rev_cnt[i] = benchmark['mean_runtime'] / 1000
        reviewer_count[i] = benchmark['reviewers']

    mean_runtimes_pap_cnt = np.zeros(len(data)).astype(np.float64)
    paper_count = np.zeros(len(data)).astype(np.uint32)
    for i, benchmark in enumerate(sorted(data, key=lambda x: x["papers"])):
        mean_runtimes_pap_cnt[i] = benchmark['mean_runtime'] / 1000
        paper_count[i] = benchmark['papers']

    mean_runtimes_product = np.zeros(len(data)).astype(np.float64)
    product_vals = np.zeros(len(data)).astype(np.uint32)
    for i, benchmark in enumerate(sorted(data, key=lambda x: x["papers"] * x["reviewers"])):
        mean_runtimes_product[i] = benchmark['mean_runtime'] / 1000
        product_vals[i] = benchmark['papers'] * benchmark['reviewers']
    
    plt.figure(figsize=(14, 4))

    #paper count 
    plt.subplot(1,3,1)
    plt.plot(paper_count, mean_runtimes_pap_cnt, marker="o")

    slope, intercept, r_value, p_value, std_err = linregress(paper_count, mean_runtimes_pap_cnt)
    regression_line = [slope * val + intercept for val in paper_count]
    plt.plot(paper_count, regression_line, color="red", label=f"Regression Line (R²={r_value**2:.3f})")
    plt.legend()

    plt.xlabel("Paper count")
    plt.ylabel("Mean runtime [s]")
    plt.title("Mean runtime per paper count")

    #review count
    plt.subplot(1,3,2)
    plt.plot(reviewer_count, mean_runtimes_rev_cnt, marker="o")

    slope, intercept, r_value, p_value, std_err = linregress(reviewer_count, mean_runtimes_rev_cnt)
    regression_line = [slope * val + intercept for val in reviewer_count]
    plt.plot(reviewer_count, regression_line, color="red", label=f"Regression Line (R²={r_value**2:.3f})")
    plt.legend()

    plt.xlabel("Reviewer count")
    plt.ylabel("Mean runtime [s]")
    plt.title("Mean runtime per reviewer count")

    #product
    plt.subplot(1,3,3)
    plt.plot(product_vals, mean_runtimes_product, marker="o")

    slope, intercept, r_value, p_value, std_err = linregress(product_vals, mean_runtimes_product)
    regression_line = [slope * val + intercept for val in product_vals]
    plt.plot(product_vals, regression_line, color="red", label=f"Regression Line (R²={r_value**2:.3f})")
    plt.legend()

    plt.xlabel("Reviewer count * Paper count")
    plt.ylabel("Mean runtime [s]")
    plt.title("MRT per (papers*reviewers) product")

    plt.show()

def compare_runtimes(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.is_absolute():
        folder_path = PROJECT_ROOT / folder_path
    benchmarks = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            data = read_benchmark_json(file_path)
            no_ext = os.path.splitext(os.path.basename(file_path))[0]
            name = no_ext.split('/')[-1]
            name = name.removesuffix('_benchmarks').replace('_', ' ')
            benchmark = (name, data)
            benchmarks.append(benchmark)
    
    plt.figure(figsize=(6, 6))

    for benchmark in benchmarks:

        mean_runtimes_product = np.zeros(len(data)).astype(np.float64)
        product_vals = np.zeros(len(data)).astype(np.uint32)
        for i, bm in enumerate(sorted(benchmark[1], key=lambda x: x["papers"] * x["reviewers"])):
            mean_runtimes_product[i] = bm['mean_runtime'] / 1000
            product_vals[i] = bm['papers'] * bm['reviewers']

        plt.plot(product_vals, mean_runtimes_product, marker="o", label=benchmark[0])
        
    plt.legend()
    plt.xlabel("Papers * Reviewers product")
    plt.ylabel("Mean runtime [s]")
    plt.title("Mean runtimes for different algorithms")

    plt.show()
    