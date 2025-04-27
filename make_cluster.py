#!/usr/bin/env python3
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np
import pandas as pd
import argparse
import os

def generate_blob(n, k, std, dim, out):
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=std, n_features=dim, random_state=42)
    pd.DataFrame(X).to_csv(out, index=False, header=False)
    print(f"Saved: {out}")

def generate_moons(n, out):
    X, _ = make_moons(n_samples=n, noise=0.1, random_state=42)
    pd.DataFrame(X).to_csv(out, index=False, header=False)
    print(f"Saved: {out}")

def generate_circles(n, out):
    X, _ = make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=42)
    pd.DataFrame(X).to_csv(out, index=False, header=False)
    print(f"Saved: {out}")

def generate_unbalanced(n1, n2, std1, std2, out):
    X1, _ = make_blobs(n_samples=n1, centers=[[0,0]], cluster_std=std1, random_state=42)
    X2, _ = make_blobs(n_samples=n2, centers=[[5,5]], cluster_std=std2, random_state=43)
    X = np.vstack([X1, X2])
    pd.DataFrame(X).to_csv(out, index=False, header=False)
    print(f"Saved: {out}")

def generate_all():
    os.makedirs("data", exist_ok=True)
    # small/debug
    generate_blob(100, 5, 0.6, 2, "data/data_debug_k5.csv")
    generate_blob(100, 5, 0.6, 3, "data/data_highdim_k5.csv")
    generate_blob(500, 3, 1.0, 2, "data/data_clear_mini_k3.csv")
    generate_blob(10000, 3, 1.0, 2, "data/data_clear_small_k3.csv")

    
    generate_blob(100000, 3, 3.0, 2, "data/data_overlap_med_k3.csv")
    generate_blob(100000, 10, 1.0, 2, "data/data_highk_k10.csv")
    generate_blob(100000, 3, 1.0, 5, "data/data_highdim_k3_5d.csv")

    # large scale (for MPI, CUDA, Hybrid)
    generate_blob(500000, 3, 3.0, 2, "data/data_overlap_xlarge_k3.csv")
    generate_blob(1000000, 3, 3.0, 2, "data/data_overlap_xxlarge_k3.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['blob', 'moons', 'circles', 'unbalanced'], help='Dataset type')
    parser.add_argument('--n', type=int, help='Number of samples')
    parser.add_argument('--k', type=int, help='Number of clusters')
    parser.add_argument('--std', type=float, default=1.0, help='Cluster std dev')
    parser.add_argument('--dim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--out', type=str, help='Output CSV filename')
    parser.add_argument('--all', action='store_true', help='Generate all test cases')

    args = parser.parse_args()

    if args.all:
        generate_all()
    elif args.type == 'blob':
        generate_blob(args.n, args.k, args.std, args.dim, args.out)
    elif args.type == 'moons':
        generate_moons(args.n, args.out)
    elif args.type == 'circles':
        generate_circles(args.n, args.out)
    elif args.type == 'unbalanced':
        generate_unbalanced(1000, 9000, 0.5, 1.5, args.out)
    else:
        print("Invalid arguments. Use --help for more info.")

if __name__ == "__main__":
    main()
