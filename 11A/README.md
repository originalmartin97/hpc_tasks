cat > README.md << 'EOT'
# Easter Egg Hunt Simulation

## Overview

This folder contains three parallel implementations of an Easter egg hunt simulation. The simulation creates a square field filled with randomly placed eggs, then uses different parallel computing techniques to search for all eggs as efficiently as possible.

## Problem Description

- A square field of size N×N (default 100×100) is populated with hidden Easter eggs
- Multiple searcher agents work together to find all eggs as quickly as possible
- Each implementation uses a different parallel programming paradigm

## Implementations

### 1. MPI Version (`eggwar_mpi.c`)
Uses the Message Passing Interface to distribute the search across multiple processes, with each process handling a portion of the field.

### 2. Hybrid MPI+OpenMP Version (`eggwar_hybrid.c`)
Combines MPI for distributing work across nodes with OpenMP for shared-memory parallelism within each node, creating two levels of parallelism.

### 3. CUDA Version (`eggwar_cuda.cu`)
Leverages GPU acceleration to process thousands of grid cells in parallel using NVIDIA's CUDA platform.

## Compilation

```bash
# MPI version
mpicc -o eggwar eggwar_mpi.c

# Hybrid version
mpicc -fopenmp -o eggwar_hybrid eggwar_hybrid.c

# CUDA version
nvcc -o eggwar_cuda eggwar_cuda.cu