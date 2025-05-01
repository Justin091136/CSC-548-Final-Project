#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math

def geometric_mean(arr):
    arr = np.array(arr)
    arr = arr[arr > 0]
    return math.exp(np.mean(np.log(arr)))

def load_times(method, test_name, variant):
    filepath = f"./results/runtime_csv/{test_name}/execution_times_{method}_{variant}.csv"
    df = pd.read_csv(filepath)
    return df.set_index('trial')['time_ms']

def compute_speedups(method, test_name):
    seq = load_times(method, test_name, 'seq')
    openmp = load_times(method, test_name, 'openmp')
    mpi = load_times(method, test_name, 'mpi')
    hybrid = load_times(method, test_name, 'hybrid')

    speedup_openmp = seq / openmp
    speedup_mpi = seq / mpi
    speedup_hybrid = seq / hybrid

    gm_openmp = geometric_mean(speedup_openmp)
    gm_mpi = geometric_mean(speedup_mpi)
    gm_hybrid = geometric_mean(speedup_hybrid)

    print(f"[{method.upper()}][{test_name}] Speedup Summary:")
    print(f"  OpenMP : GeoMean = {gm_openmp:.4f}")
    print(f"  MPI    : GeoMean = {gm_mpi:.4f}")
    print(f"  Hybrid : GeoMean = {gm_hybrid:.4f}\n")

if __name__ == "__main__":
    methods = ['kmeans', 'gmm']
    test_files = [
        "data_overlap_med_k3",
        "data_highk_k10",
        "data_highdim_k3_5d"
    ]

    for method in methods:
        for test in test_files:
            compute_speedups(method, test)
        print("-------------------------------")
