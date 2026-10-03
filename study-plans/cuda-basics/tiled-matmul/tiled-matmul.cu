#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float shared_A[TILE_DIM][TILE_DIM];
    __shared__ float shared_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;
    float Cvalue = 0.0f;
      int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int m = 0; m < num_tiles; ++m)
    {
        int A_col = m * TILE_DIM + tx;
        int B_row = m * TILE_DIM + ty;

        if (row < M && A_col < K)
            shared_A[ty][tx] = A[row * K + A_col];
        else
            shared_A[ty][tx] = 0.0f;
        if (B_row < K && col < N)
            shared_B[ty][tx] = B[B_row * N + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k)
        {
            Cvalue += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = Cvalue;
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}