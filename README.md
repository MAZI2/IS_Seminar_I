# Conference paper assignment

Genetic algorithms for assigning conference papers to reviewers, with capacity, authorship, and friendship constraints.

```bash
python -m pip install -r requirements.txt
jupyter notebook report.ipynb
```

Run the notebook from the repository root. `datasets/` contains eight inputs; `benchmarks/` contains the saved results.

To rerun all benchmarks:

```bash
python algorithms/benchmark_all_models.py
```

The benchmark script runs five algorithms with 30 repetitions per dataset, so this can take a while.
