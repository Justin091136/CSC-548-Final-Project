# Final Project - CSC 548 Parallel Systems

This repository contains the final project for NCSU CSC 548.

## Building

To build all binaries, simply run:

    make

The binaries will be generated automatically in the `bin/` directory.

The available binaries are:

- K-Means:
  - `kmeans`
  - `kmeans-openmp`
  - `kmeans-mpi`
  - `kmeans-hybrid`
- GMM:
  - `gmm`
  - `gmm-openmp`
  - `gmm-mpi`
  - `gmm-hybrid`

The Makefiles support both Linux and macOS.  
On macOS, the Makefiles use LLVM's `clang++` with OpenMP support.

## Usage

Run the binary with the input CSV file:

    <binary_file> <test_file.csv>

Example (running hybrid K-Means with 4 MPI processes):

    mpirun -np 4 ./bin/kmeans-hybrid ./data/data_highk_k10.csv

## Input File Format

Input files must follow the naming convention:

    *_k<cluster_count>_*.csv

For example:

- `data_overlap_med_k3` means the dataset contains 3 clusters (k = 3).
- `data_highk_k10.csv` indicates the dataset contains 10 clusters (k = 10).

