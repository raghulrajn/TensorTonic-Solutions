#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][TILE_DIM][TILE_DIM];
    __shared__ float Bs[2][TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int buf = 0;
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;
    float acc = 0.0f;
      int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

   int A_col = tx;
    int B_row = ty;

    if (row < M && A_col < K)
        As[0][ty][tx] = A[row * K + A_col];
    else
        As[0][ty][tx] = 0.0f;

    if (B_row < K && col < N)
        Bs[0][ty][tx] = B[B_row * N + col];
    else
        Bs[0][ty][tx] = 0.0f;

    __syncthreads();

    for (int t = 0; t < num_tiles; ++t)
    {
        int next = buf ^ 1;
        if (t + 1 < num_tiles)
        {
            int next_A_col = (t + 1) * TILE_DIM + tx;
            int next_B_row = (t + 1) * TILE_DIM + ty;

            if (row < M && next_A_col < K)
            {
                As[next][ty][tx] =A[row * K + next_A_col];
            }
            else
            {
                As[next][ty][tx] = 0.0f;
            }

            if (next_B_row < K && col < N)
            {
                Bs[next][ty][tx] = B[next_B_row * N + col];
            }
            else
            {
                Bs[next][ty][tx] = 0.0f;
            }
        }

        for (int k = 0; k < TILE_DIM; ++k)
        {
            acc +=As[buf][ty][k] *Bs[buf][k][tx];
        }

        __syncthreads();

        buf = next;
    }
    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}