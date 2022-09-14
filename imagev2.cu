#include <cuda.h>
#include <cooperative_groups.h>
#include <random>
#include <cstdio>

__global__
void gpu_transpose(
    float * a,
    int  dim1,
    int  dim2,
    int  dim3,
    int  dim4
) {
      int c = threadIdx.x;
      int h = blockIdx.x;
      int w = blockIdx.y;
      float tmp;
      int start = (h<w) ? 0 : dim1/2;
      int end = (h<w) ? dim1/2 : dim1;
   
      if(h!=w) {
        for (int n = start; n < end; n++) {
          tmp = a [n*dim2*dim3*dim4 + h*dim3*dim4 + w*dim4 + dim4/2 + c];
          a [n*dim2*dim3*dim4 + h*dim3*dim4 + w*dim4 + dim4/2 + c] = a [n*dim2*dim3*dim4 + w*dim3*dim4 + h*dim4 + dim4/2 + c];
          a [n*dim2*dim3*dim4 + w*dim3*dim4 + h*dim4 + dim4/2 + c] = tmp;
        }
      }
}

void cpu_transpose(
    float * a,
    float * b,
    int  dim1,
    int  dim2,
    int  dim3,
    int  dim4
) {
    for(int i=0; i<dim1; i++)
      for(int j=0; j<dim2; j++)
        for(int k=0; k<dim3; k++)
          for(int w=0; w<dim4; w++)
              if(w<dim4/2)
                b[i*dim2*dim3*dim4 + j*dim3*dim4 + k*dim4 + w] = a[i*dim2*dim3*dim4 + j*dim3*dim4 + k*dim4 + w];
              else
                b[i*dim2*dim3*dim4 + j*dim3*dim4 + k*dim4 + w] = a[i*dim2*dim3*dim4 + k*dim3*dim4 + j*dim4 + w];
}

void random_fill(float * data, int num_elems) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    for (int i = 0; i < num_elems; i++) {
        data[i] = dis(gen);
    }
}

int main(int argc, char ** argv) {
    constexpr int N = 4;
    constexpr int H = 32;
    constexpr int W = 32;
    constexpr int C = 1024;
    constexpr int C_by_2 = C/2;

    float * x, * cpu_result;
    float * x_device, * gpu_result;
    constexpr int x_size = sizeof(float) * N * H * W * C;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocHost(&x,  x_size);
    cudaMallocHost(&cpu_result, x_size);
    cudaMallocHost(&gpu_result, x_size);
  
    random_fill(x, N * H * W * C);

    cudaMalloc(&x_device, x_size);
    cudaMemcpyAsync(x_device, x, x_size, cudaMemcpyHostToDevice, stream);

    dim3 block(C_by_2);
    dim3 grid (H, W);
              
    gpu_transpose<<<grid, block, 0, stream>>>(x_device, N, H, W, C);

    cudaMemcpyAsync(gpu_result, x_device, x_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int mismatch = 0;

    cpu_transpose(x, cpu_result, N, H, W, C);

    for(int i=0; i<N; i++)
      for(int j=0; j<H; j++)
        for(int k=0; k<W; k++)
          for(int w=0; w<C; w++)
            if (gpu_result[i*H*W*C+ j*W*C + k*C + w] != cpu_result[i*H*W*C+ j*W*C + k*C + w]) {
				mismatch++;
			}

    printf("mismatch: %d\n", mismatch);
    

    // Clean up
    cudaFree(x_device);
    cudaFreeHost(x);
    cudaFreeHost(gpu_result);
    cudaFreeHost(cpu_result);

    cudaStreamDestroy(stream);

    return 0;
}