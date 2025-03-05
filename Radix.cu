#include "cuda_runtime.h"
#include <iostream>


#define Radix 4  
#define bits 10  

__global__ void Count(int* input, int* output, int shift, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        int bucket = (input[i] >> shift) & (Radix - 1);
        atomicAdd(&output[bucket], 1);
    }
}

__global__ void scan(int* inp, int* out) {
    __shared__ int temp[Radix];

    int x = threadIdx.x;
    if (x < Radix) {
        temp[x] = inp[x];
    }
    __syncthreads();

    
    int val = 0;
    for (int i = 1; i < Radix; i <<= 1) {
        if (x >= i) val = temp[x - i];
        __syncthreads();
        if (x >= i) temp[x] += val;
        __syncthreads();
    }

    if (x < Radix) {
        out[x] = (x == 0) ? 0 : temp[x - 1]; 
    }
}

__global__ void rearrange(int* inp, int* out, int* scan, int shift, int n, int* temp_indices) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < n) {
        int key = (inp[x] >> shift) & (Radix - 1);
        int newidx = atomicAdd(&scan[key], 1);
        out[newidx] = inp[x];
        temp_indices[newidx] = x; 
    }
}

void radix_sort(int* input, int len) {
    int *i_d, *o_d, *scan_d, *count, *temp_indices;

    int* temp = (int*)malloc(len * sizeof(int));

    cudaMalloc(&i_d, sizeof(int) * len);
    cudaMalloc(&o_d, sizeof(int) * len);
    cudaMalloc(&scan_d, sizeof(int) * Radix);
    cudaMalloc(&count, sizeof(int) * Radix);
    cudaMalloc(&temp_indices, sizeof(int) * len);

    cudaMemcpy(i_d, input, len * sizeof(int), cudaMemcpyHostToDevice);

    for (int pass = 0; pass < bits / 2; pass++) {
        int shift = pass * 2;

        cudaMemset(count, 0, Radix * sizeof(int));

        Count<<<(len + 255) / 256, 256>>>(i_d, count, shift, len);
        cudaDeviceSynchronize();

        scan<<<1, Radix>>>(count, scan_d);
        cudaDeviceSynchronize();

        rearrange<<<(len + 255) / 256, 256>>>(i_d, o_d, scan_d, shift, len, temp_indices);
        cudaDeviceSynchronize();

        cudaMemcpy(temp, o_d, len * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Pass " << pass + 1 << ": ";
        for (int i = 0; i < len; i++) std::cout << temp[i] << " ";
        std::cout << std::endl;

        cudaMemcpy(i_d, o_d, len * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(input, i_d, len * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(i_d);
    cudaFree(o_d);
    cudaFree(scan_d);
    cudaFree(count);
    cudaFree(temp_indices);
    free(temp);
}

int main() {
    int h_input[] = {512, 768, 256, 1023, 100, 10, 700, 50};
    int n = sizeof(h_input) / sizeof(h_input[0]);

    radix_sort(h_input, n);

    std::cout << "Sorted Array: ";
    for (int i = 0; i < n; i++) std::cout << h_input[i] << " ";
    std::cout << std::endl;

    return 0;
}
