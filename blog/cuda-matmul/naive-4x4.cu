// Build with the toolkit and GPU target selected in the visualization.
// Example for A100: nvcc -std=c++14 -O3 -lineinfo -arch=sm_80 naive-4x4.cu -o naive-4x4
// Profile: ncu --metrics gpc__cycles_elapsed.max,gpu__time_duration.sum --print-units base ./naive-4x4
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

// Original kernel from https://siboehm.com/articles/22/CUDA-MMM
/*
MIT License

Copyright (c) 2023 Simon Boehm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Kernel 2 uses the body shown in the article, with a one-dimensional block.
constexpr int BLOCKSIZE = 32;
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
                                 const float *B, float beta, float *C) {
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Kernel 3 specializes the article's shared-memory kernel for complete tiles.
template <int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B, float beta, float *C) {
  const uint cRow = blockIdx.x, cCol = blockIdx.y;
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  // the outer loop advances A along the columns and B along
  // the rows until we have fully calculated the result in C.
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B from
    // global memory into shared memory.
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
              Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

// Kernel 4 from Simon Böhm’s MIT-licensed SGEMM_CUDA implementation.
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && BM % TM == 0,
                "A thread strip must fit the block");
  constexpr int threads = (BM / TM) * BN;
  static_assert(threads <= 1024, "Too many threads for one block");
  const int t = threadIdx.x;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // The blog's tile shape assigns one word of each input to each thread.
    if (BM * BK == threads && BK * BN == threads) {
      As[t] = A[(t / BK) * K + t % BK];
      Bs[t] = B[(t / BN) * N + t % BN];
    } else {
      for (int w = t; w < BM * BK; w += threads)
        As[w] = A[(w / BK) * K + w % BK];
      for (int w = t; w < BK * BN; w += threads)
        Bs[w] = B[(w / BN) * N + w % BN];
    }
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// Adapted from Simon Böhm’s Kernel 5, with a row-major shared-memory epilogue.
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // A thread is responsible for calculating TM*TN elements in the blocktile
  static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0 && BM % TM == 0 && BN % TN == 0,
                "Each block must contain complete thread tiles");
  constexpr int numThreadsBlocktile = (BM / TM) * (BN / TN);
  static_assert(numThreadsBlocktile <= 1024, "Too many threads for one block");

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    if (numThreadsBlocktile % BK == 0 && numThreadsBlocktile % BN == 0 &&
        (BM * BK) % numThreadsBlocktile == 0 && (BK * BN) % numThreadsBlocktile == 0) {
      for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
      }
      for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
      }
    } else {
      for (int w = threadIdx.x; w < BM * BK; w += numThreadsBlocktile)
        As[w] = A[(w / BK) * K + w % BK];
      for (int w = threadIdx.x; w < BK * BN; w += numThreadsBlocktile)
        Bs[w] = B[(w / BN) * N + w % BN];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // Stage the finished register tiles in shared memory.
  __shared__ float Cs[BM * BN];
  for (int r = 0; r < TM; ++r) {
    for (int c = 0; c < TN; ++c) {
      Cs[(threadRow * TM + r) * BN + threadCol * TN + c] = threadResults[r * TN + c];
    }
  }
  __syncthreads();

  // Neighboring lanes write neighboring words of the tile.
  const int warpBase = (threadIdx.x / warpSize) * warpSize;
  const int writers = min(warpSize, numThreadsBlocktile - warpBase);
  for (int pass = 0; pass < TM * TN; ++pass) {
    const int w = warpBase * TM * TN + pass * writers + threadIdx.x % warpSize;
    const int idx = (w / BN) * N + w % BN;
    C[idx] = alpha * Cs[w] + beta * C[idx];
  }
}

// Adapted from Simon Böhm's Kernel 6. Strided packet loads also cover smaller teaching tiles.
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  static_assert(TM > 0 && TN > 0 && BM % TM == 0 && BN % TN == 0,
                "Output tiles must contain complete thread tiles");
  static_assert(BK % 4 == 0 && TN % 4 == 0 && BN % 4 == 0,
                "Input and output packets require four aligned floats");
  constexpr int blockThreads = (BM / TM) * (BN / TN);
  static_assert(blockThreads <= 1024, "Too many threads in the block");
  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  __shared__ __align__(16) float As[BK * BM];
  __shared__ __align__(16) float Bs[BK * BN];
  float threadResults[TM * TN] = {};
  float regM[TM], regN[TN];

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Each participating lane loads four adjacent floats at once.
    for (int w = 4 * threadIdx.x; w < BM * BK; w += 4 * blockThreads) {
      const int row = w / BK, col = w % BK;
      const float4 packet = *reinterpret_cast<const float4 *>(&A[row * K + col]);
      As[(col + 0) * BM + row] = packet.x;
      As[(col + 1) * BM + row] = packet.y;
      As[(col + 2) * BM + row] = packet.z;
      As[(col + 3) * BM + row] = packet.w;
    }
    for (int w = 4 * threadIdx.x; w < BK * BN; w += 4 * blockThreads) {
      *reinterpret_cast<float4 *>(&Bs[w]) =
          *reinterpret_cast<const float4 *>(&B[(w / BN) * N + w % BN]);
    }
    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i)
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      for (int i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN)
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }

  // Write C directly in four-float packets.
  for (int r = 0; r < TM; ++r) {
    for (int c = 0; c < TN; c += 4) {
      const int idx = (threadRow * TM + r) * N + threadCol * TN + c;
      float4 old = *reinterpret_cast<const float4 *>(&C[idx]);
      old.x = alpha * threadResults[r * TN + c + 0] + beta * old.x;
      old.y = alpha * threadResults[r * TN + c + 1] + beta * old.y;
      old.z = alpha * threadResults[r * TN + c + 2] + beta * old.z;
      old.w = alpha * threadResults[r * TN + c + 3] + beta * old.w;
      *reinterpret_cast<float4 *>(&C[idx]) = old;
    }
  }
}

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

constexpr bool powerOfTwo(long value) {
  return value > 0 && (value & (value - 1)) == 0;
}

static int dimension(const char *arg, int upper) {
  char *end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (*arg == '\0' || *end != '\0' || value > upper || !powerOfTwo(value)) {
    std::fprintf(stderr, "Expected a power-of-two dimension from 1 to %d.\n", upper);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

static std::vector<float> sequentialMatrix(int rows, int cols) {
  std::vector<float> values(rows * cols);
  std::iota(values.begin(), values.end(), 1.0f);
  return values;
}

static std::vector<float> initialC(int rows, int cols) {
  constexpr float seed[]={1,2,1,2, 2,1,2,1, 1,1,2,2, 2,2,1,1};
  std::vector<float> values(rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      values[r * cols + c] = seed[(r % 4) * 4 + c % 4];
  return values;
}

template <int Side>
static void launchShared(int m, int k, int n, float *a, float *b, float *c) {
  sgemm_shared_mem_block<Side><<<dim3(m / Side, n / Side), Side * Side>>>(
      m, n, k, 2.0f, a, b, 3.0f, c);
}

// Each entry launches the same compile-time BLOCKSIZE specialization shown in the trace.
template <std::size_t... I>
static void launchSharedSize(std::index_sequence<I...>, int side, int m, int k, int n,
                             float *a, float *b, float *c) {
  using Launcher = void (*)(int, int, int, float *, float *, float *);
  static constexpr Launcher launchers[] = {launchShared<I + 1>...};
  launchers[side - 1](m, k, n, a, b, c);
}

// The visualizer's profiler command sets these template dimensions at build time.
#ifndef K4_BM
#define K4_BM 4
#endif
#ifndef K4_BK
#define K4_BK 2
#endif
#ifndef K4_BN
#define K4_BN 4
#endif
#ifndef K4_TM
#define K4_TM 2
#endif

#ifndef K5_BM
#define K5_BM 4
#endif
#ifndef K5_BK
#define K5_BK 2
#endif
#ifndef K5_BN
#define K5_BN 4
#endif
#ifndef K5_TM
#define K5_TM 2
#endif
#ifndef K5_TN
#define K5_TN 2
#endif

#ifndef K6_BM
#define K6_BM 16
#endif
#ifndef K6_BK
#define K6_BK 4
#endif
#ifndef K6_BN
#define K6_BN 16
#endif
#ifndef K6_TM
#define K6_TM 4
#endif
#ifndef K6_TN
#define K6_TN 4
#endif

static_assert(powerOfTwo(K4_BM) && powerOfTwo(K4_BK) && powerOfTwo(K4_BN) && powerOfTwo(K4_TM),
              "Kernel 4 tile dimensions must be powers of two");
static_assert(powerOfTwo(K5_BM) && powerOfTwo(K5_BK) && powerOfTwo(K5_BN) &&
              powerOfTwo(K5_TM) && powerOfTwo(K5_TN),
              "Kernel 5 tile dimensions must be powers of two");

static_assert(powerOfTwo(K6_BM) && powerOfTwo(K6_BK) && powerOfTwo(K6_BN) &&
              powerOfTwo(K6_TM) && powerOfTwo(K6_TN),
              "Kernel 6 tile dimensions must be powers of two");

int main(int argc, char **argv) {
  const bool coalesced = argc == 5 && std::strcmp(argv[4], "coalesced") == 0;
  const bool shared = (argc == 5 || argc == 6) && std::strcmp(argv[4], "shared") == 0;
  const bool registerTiled = argc == 5 && std::strcmp(argv[4], "register") == 0;
  const bool register2D = argc == 5 && std::strcmp(argv[4], "register2d") == 0;
  const bool vectorized = argc == 5 && std::strcmp(argv[4], "vectorized") == 0;
  if (argc != 1 && argc != 4 && !coalesced && !shared && !registerTiled && !register2D && !vectorized) {
    std::fprintf(stderr, "Usage: %s [M K N [coalesced|shared [block-side]|register|register2d|vectorized]]\n", argv[0]);
    return EXIT_FAILURE;
  }
  constexpr int maxDimension = 1024;
  const int m = argc >= 4 ? dimension(argv[1], maxDimension) : 4;
  const int k = argc >= 4 ? dimension(argv[2], maxDimension) : 4;
  const int n = argc >= 4 ? dimension(argv[3], maxDimension) : 4;
  constexpr int tile = 4;
  const int side = shared && argc == 6 ? dimension(argv[5], BLOCKSIZE) : tile;
  const int height = registerTiled ? K4_BM : register2D ? K5_BM : vectorized ? K6_BM : side;
  const int width = registerTiled ? K4_BN : register2D ? K5_BN : vectorized ? K6_BN : side;
  const int depth = registerTiled ? K4_BK : register2D ? K5_BK : vectorized ? K6_BK : side;
  if ((shared || registerTiled || register2D || vectorized) && (m % height || k % depth || n % width)) {
    std::fprintf(stderr, "M, K and N must be multiples of %d, %d and %d.\n", height, depth, width);
    return EXIT_FAILURE;
  }
  const auto A = sequentialMatrix(m, k);
  const auto B = sequentialMatrix(k, n);
  const auto oldC = initialC(m, n);
  std::vector<float> result(m * n);
  const auto bytesA = A.size() * sizeof(float), bytesB = B.size() * sizeof(float);
  const auto bytesC = oldC.size() * sizeof(float);
  float *deviceA, *deviceB, *deviceC;
  check(cudaMalloc(&deviceA, bytesA));
  check(cudaMalloc(&deviceB, bytesB));
  check(cudaMalloc(&deviceC, bytesC));
  check(cudaMemcpy(deviceA, A.data(), bytesA, cudaMemcpyHostToDevice));
  check(cudaMemcpy(deviceB, B.data(), bytesB, cudaMemcpyHostToDevice));
  check(cudaMemcpy(deviceC, oldC.data(), bytesC, cudaMemcpyHostToDevice));

  const dim3 globalGrid((m + BLOCKSIZE - 1) / BLOCKSIZE, (n + BLOCKSIZE - 1) / BLOCKSIZE);
  if (vectorized) {
    sgemmVectorize<K6_BM, K6_BN, K6_BK, K6_TM, K6_TN>
        <<<dim3(n / K6_BN, m / K6_BM), (K6_BM / K6_TM) * (K6_BN / K6_TN)>>>(
        m, n, k, 2.0f, deviceA, deviceB, 3.0f, deviceC);
  } else if (register2D) {
    sgemm2DBlocktiling<K5_BM, K5_BN, K5_BK, K5_TM, K5_TN>
        <<<dim3(n / K5_BN, m / K5_BM), (K5_BM / K5_TM) * (K5_BN / K5_TN)>>>(
        m, n, k, 2.0f, deviceA, deviceB, 3.0f, deviceC);
  } else if (registerTiled) {
    sgemm1DBlocktiling<K4_BM, K4_BN, K4_BK, K4_TM>
        <<<dim3(n / K4_BN, m / K4_BM), (K4_BM / K4_TM) * K4_BN>>>(
        m, n, k, 2.0f, deviceA, deviceB, 3.0f, deviceC);
  } else if (shared) {
    launchSharedSize(std::make_index_sequence<BLOCKSIZE>{}, side, m, k, n, deviceA, deviceB, deviceC);
  } else if (coalesced) {
    sgemm_coalescing<<<globalGrid, BLOCKSIZE * BLOCKSIZE>>>(m, n, k, 2.0f, deviceA, deviceB, 3.0f, deviceC);
  } else {
    sgemm_naive<<<globalGrid, dim3(BLOCKSIZE, BLOCKSIZE)>>>(m, n, k, 2.0f, deviceA, deviceB, 3.0f, deviceC);
  }
  check(cudaGetLastError());
  check(cudaMemcpy(result.data(), deviceC, bytesC, cudaMemcpyDeviceToHost));
  check(cudaFree(deviceA));
  check(cudaFree(deviceB));
  check(cudaFree(deviceC));

  const bool printCells = m <= 32 && n <= 32;
  std::vector<float> sums(n);
  for (int r = 0; r < m; ++r) {
    std::fill(sums.begin(), sums.end(), 0.0f);
    for (int i = 0; i < k; ++i)
      for (int c = 0; c < n; ++c)
        sums[c] = std::fma(A[r*k + i], B[i*n + c], sums[c]);
    for (int c = 0; c < n; ++c) {
      const int index = r*n + c;
      if (result[index] != std::fma(2.0f, sums[c], 3.0f * oldC[index])) {
        std::fprintf(stderr, "Unexpected result at row %d, column %d\n", r, c);
        return EXIT_FAILURE;
      }
      if (printCells) std::printf("%8.0f", result[index]);
    }
    if (printCells) std::printf("\n");
  }
  if (!printCells) std::printf("Verified all %d × %d outputs.\n", m, n);
}
