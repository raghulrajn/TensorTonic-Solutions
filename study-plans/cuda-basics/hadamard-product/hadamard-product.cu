#include <cuda_runtime.h>

__global__ void hadamard_kernel(const float* A, const float* B, float* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row<=M && col <=N){
        int idx = row*N+col;
        C[idx] = A[idx]*B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hadamard_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
