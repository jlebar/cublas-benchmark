# A Simple Utility for Benchmarking CUBLAS

## Compiling and Running

```
$ /usr/local/cuda-X.0/bin/nvcc --std=c++11 -arch=compute_60 -code=sm_60 main.cu \
  -lcublas -o matmul
$ LD_LIBRARY_PATH=/usr/local/cuda-X.0/lib64 ./matmul M N K TA TB
```

Where `M`, `N`, `K` are integers defining the size of the matrices to be
multiplied, and `TA` and `TB` are integers 0 or 1 indicating whether the A and B
matrices should be transposed.

If you have multiple nvidia GPUs in your machine, set the `CUDA_VISIBLE_DEVICES`
env var to the appropriate number before running the benchmark.

## Running the Benchmark Suite

```
$ cat benchmark_args | xargs -n1 -I{} sh -c './matmul {}'
```

## Creating Pretty Output for Comparing CUDA 8 vs CUDA 9

TODO.  :)

## Disclaimer

This is not an official Google project.
