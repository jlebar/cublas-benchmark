// Simple profiling program for cublas matmul kernels.
//
// Compile with e.g.
//
//   nvcc --std=c++11 -arch=compute_60 -code=sm_60 main.cu -lcublas -o matmul
//
// Then run as e.g.
//
//   $ matmul 10 20 30 0 1
//
// which means, m=10 k=20 n=30, transpose matrix A but not B.

#include <cublas_v2.h>
#include <cassert>
#include <cstdio>

// A simple GPU Timer taken from CUB
struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void PrintDevices() {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
  }
}

void RunTest(int m, int k, int n, int transa, int transb) {
  PrintDevices();

  // Prepare on device matrix data
  const int A_rows = transa ? m : k;
  const int A_cols = transa ? k : m;
  const int B_rows = transb ? k : n;
  const int B_cols = transb ? n : k;
  const int C_rows = n;
  const int C_cols = m;

  GpuTimer gpu_timer;

  float elapsed_millis;
  float throughput;

  cublasHandle_t handle;
  cublasCreate(&handle);

  float *host_A =
      reinterpret_cast<float *>(malloc(A_rows * A_cols * sizeof(float)));
  float *host_B =
      reinterpret_cast<float *>(malloc(B_rows * B_cols * sizeof(float)));
  for (int i = 0; i < A_rows * A_cols; i++) host_A[i] = i % 100;
  for (int i = 0; i < B_rows * B_cols; i++) host_B[i] = i % 100;

  float *device_A_float, *device_B_float, *device_C_float;
  cudaMalloc(&device_A_float, A_rows * A_cols * sizeof(float));
  cudaMalloc(&device_B_float, B_rows * B_cols * sizeof(float));
  cudaMalloc(&device_C_float, C_rows * C_cols * sizeof(float));

  cudaMemcpy(device_A_float, host_A, A_rows * A_cols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_B_float, host_B, B_rows * B_cols * sizeof(float),
             cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;
  int num_iter = 100;
  printf("m:%5d,k:%5d,n:%5d, transa:%d, transb:%d\n", m, k, n, transa, transb);
  printf("Using cublasGemmEx():\n");
  // Use cublasGemmEx
  cublasGemmAlgo_t algos[] = {
    CUBLAS_GEMM_DFALT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
#if __CUDACC_VER_MAJOR__ >= 9
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
#endif
  };
  for (int i = 0; i < sizeof(algos) / sizeof(algos[0]); ++i) {
    gpu_timer.Start();
    bool result_valid = true;
    int error_code = 0;
    for (int ii = 0; ii < num_iter; ++ii) {
      auto result =
          cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                       (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                       device_B_float, CUDA_R_32F, (transb ? k : n),
                       device_A_float, CUDA_R_32F, (transa ? m : k), &beta,
                       device_C_float, CUDA_R_32F, n, CUDA_R_32F, algos[i]);
      if (result != 0) {
        result_valid = false;
        error_code = result;
        break;
      }
    }
    gpu_timer.Stop();
    if (result_valid) {
      elapsed_millis = gpu_timer.ElapsedMillis() / num_iter;
      throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
      printf(
          "algorithm:%d, runtime (msec):%6.4f, throughput "
          "(Gitems/sec):%5.2f\n",
          i, elapsed_millis, throughput);
    } else {
      printf("algorithm:%d returned error code:%d\n", i, error_code);
    }
  }
  printf("Using cublasSgemm():\n");
  gpu_timer.Start();
  bool result_valid = true;
  bool error_code = 0;
  for (int i = 0; i < num_iter; ++i) {
    auto result =
        cublasSgemm(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                    (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                    device_B_float, (transb ? k : n), device_A_float,
                    (transa ? m : k), &beta, device_C_float, n);
    if (result != 0) {
        result_valid = false;
        error_code = result;
        break;
    }
  }
  gpu_timer.Stop();
  if (result_valid) {
      elapsed_millis = gpu_timer.ElapsedMillis() / num_iter;
      throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
      printf("runtime (msec):%6.4f, throughput (Gitems/sec):%5.2f\n",
              elapsed_millis, throughput);
  } else {
      printf("cublasSgemm() returned error code:%d\n", error_code);
  }
  gpu_timer.Start();
  result_valid = false;
  error_code = 0;
  if (m == 1 && n > 1) {
      printf("Using cublasSgemv():\n");
      result_valid = true;
      for (int i = 0; i < num_iter; ++i) {
          auto result =
              cublasSgemv(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      n, k, &alpha, device_B_float, (transb ? k : n),
                      device_A_float, /*incx=*/1, &beta, device_C_float, /*incy=*/1);
          if (result != 0) {
              result_valid = false;
              error_code = result;
              break;
          }
      }
  }
  if (n == 1 && m > 1) {
      printf("Using cublasSgemv():\n");
      result_valid = true;
      for (int i = 0; i < num_iter; ++i) {
          auto result =
              cublasSgemv(handle, (transa ? CUBLAS_OP_N : CUBLAS_OP_T),
                      m, k, &alpha, device_A_float, (transa ? k : m),
                      device_B_float, /*incx=*/1, &beta, device_C_float, /*incy=*/1);
          if (result != 0) {
              result_valid = false;
              error_code = result;
              break;
          }
      }
  }
  gpu_timer.Stop();
  if (result_valid) {
      elapsed_millis = gpu_timer.ElapsedMillis() / num_iter;
      throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
      printf("runtime (msec):%6.4f, throughput (Gitems/sec):%5.2f\n",
              elapsed_millis, throughput);
  } else if (error_code != 0) {
      printf("cublasSgemv() returned error code:%d\n", error_code);
  }

  cudaFree(device_A_float);
  cudaFree(device_B_float);
  cudaFree(device_C_float);
  cublasDestroy(handle);
  free(host_A);
  free(host_B);
}

int main(int argc, char *argv[]) {
  int m, k, n, ta, tb;
  if (argc < 6) {
    // m, k, n, ta, tb
    m = 20;
    k = 20000;
    n = 200;
    ta = 0;
    tb = 1;
  } else {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    ta = atoi(argv[4]);
    tb = atoi(argv[5]);
  }
  RunTest(m, k, n, ta, tb);
  return 0;
}
