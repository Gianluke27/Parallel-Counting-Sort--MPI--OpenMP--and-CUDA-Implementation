# Parallel Counting Sort: MPI, OpenMP, and CUDA Implementation
Overview:

This repository hosts an optimized implementation of the counting sort algorithm, parallelized using MPI (Message Passing Interface), OpenMP, and CUDA. Counting sort, known for its linear-time complexity, is efficiently parallelized across distributed memory, shared memory, and GPU architectures, showcasing the versatility and performance gains achievable through parallel computing paradigms.

Features:

MPI Implementation: Utilizes MPI to distribute sorting tasks across multiple nodes in a cluster, enabling efficient sorting of large datasets through parallelization of computation and communication.

OpenMP Parallelization: Leverages OpenMP directives to parallelize sorting tasks across multiple threads within a shared-memory environment, exploiting multicore processors for enhanced performance.

CUDA Acceleration: Harnesses the computational power of NVIDIA GPUs through CUDA programming, offloading sorting tasks to the GPU for massively parallel execution, resulting in significant speedup for large-scale sorting operations.

Usage:

MPI Implementation:

Compile the MPI version of counting sort using a suitable MPI compiler.
Execute the compiled binary on a cluster with MPI support to distribute sorting tasks across multiple nodes.
OpenMP Parallelization:

Compile the OpenMP version of counting sort using a compatible compiler with OpenMP support.
Adjust the number of threads via environment variables or compiler directives to optimize performance for your system configuration.
CUDA Acceleration:

Compile the CUDA version of counting sort using the NVIDIA CUDA toolkit.
Ensure proper configuration of CUDA runtime settings and device selection for optimal GPU utilization.