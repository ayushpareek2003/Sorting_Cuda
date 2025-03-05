#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 256
#define Radix 8


__global__ void scan(unsigned int* inp){
    __shared__ int temp[Radix];

    int x=threadIdx.x;
    
    if(x==0){
        temp[x]=0;
    }
    else{
        temp[x]=inp[x-1];
    }

    __syncthreads();

    for(int i=1;i<Radix;i*=2){
        
        int val=0;
        if(x>=i){
            val=temp[x-i];
        }
        __syncthreads();

        if(x>=i){
            temp[x]+=val;
        }
        __syncthreads();

    }
    
    if(x<Radix){
       
        inp[x]=temp[x];
    }

}

__global__ void exclusiveScan(unsigned int* bits, unsigned int N) {
    __shared__ unsigned int temp[BLOCK_SIZE];
    int i = threadIdx.x;
    
    if (i < N) temp[i] = bits[i];
    else temp[i] = 0;
    __syncthreads();
    
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        unsigned int val = 0;
        if (i >= offset) val = temp[i - offset];
        __syncthreads();
        if (i >= offset) temp[i] += val;
        __syncthreads();
    }
    
    if (i < N) bits[i] = temp[i];
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output,
                               unsigned int* bits, unsigned int N, unsigned int iter) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
    __syncthreads();

    // exclusiveScan(bits, N);
    __syncthreads();
    
    if (i < N) {
        unsigned int numOnesTotal = bits[N - 1] + ((input[N - 1] >> iter) & 1);
        unsigned int numOnesBefore = bits[i];
        unsigned int dst = (bits[i] == 0) ? (i - numOnesBefore)  
                                          : (N - numOnesTotal + numOnesBefore);
        output[dst] = input[i];
    }
}

void radix_sort(unsigned int* h_input, unsigned int N) {
    unsigned int *d_input, *d_output, *d_bits;
    
    cudaMalloc(&d_input, N * sizeof(unsigned int));
    cudaMalloc(&d_output, N * sizeof(unsigned int));
    cudaMalloc(&d_bits, N * sizeof(unsigned int));
    
    cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (unsigned int iter = 0; iter < sizeof(unsigned int) * ; iter++) {
        radix_sort_iter<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_bits, N, iter);
        cudaDeviceSynchronize();
        std::swap(d_input, d_output); // Swap input and output for next iteration
    }
    
    cudaMemcpy(h_input, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bits);
}

int main() {
    unsigned int h_input[] = {512, 768, 256, 1023, 100, 10, 700, 50};
    unsigned int N = sizeof(h_input) / sizeof(h_input[0]);

    radix_sort(h_input, N);

    std::cout << "Sorted Array: ";
    for (unsigned int i = 0; i < N; i++) std::cout << h_input[i] << " ";
    std::cout << std::endl;

    return 0;

    
}

