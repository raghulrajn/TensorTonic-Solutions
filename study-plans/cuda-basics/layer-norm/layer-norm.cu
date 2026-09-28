#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sqsum_shared = shared + blockDim.x;

    float sum = 0.0f;
    float sqsum = 0.0f;

    const float* row_input = input + row * N;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = row_input[i];
        sum += x;
        sqsum += x * x;
    }

    sum_shared[tid] = sum;
    sqsum_shared[tid] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
            sqsum_shared[tid] += sqsum_shared[tid + stride];
        }
        __syncthreads();
    }

    float mean = sum_shared[0] / (float)N;
    float variance = sqsum_shared[0] / (float)N - mean * mean;

    variance = fmaxf(variance, 0.0f);

    float inv_std = rsqrtf(variance + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float x = row_input[i];
        float normalized = (x - mean) * inv_std;
        output[row * N + i] = normalized * gamma[i] + beta[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shmem = 2 * threads * sizeof(float);
    layer_norm_kernel<<<blocks, threads,shmem>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
