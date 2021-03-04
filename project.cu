#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "viscous.h"
#include <cuda.h>
#include "../matplotlib-cpp-master/matplotlibcpp.h"
#include "kernel.h"

#define GRID_SIZE 512
#define BLOCK_SIZE 16

namespace plt = matplotlibcpp;

int parity(int di, int dj, int i, int j, int rho);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char *argv[]){

	int nx = 512;
	int ny = 512;
	float h = 1.0f/nx ;
	int size = nx*ny;
	int size_x = (nx+1)*(ny);
	int size_y = nx*(ny+1);
  int size_3D = 3*size;

	// memory allocation
	float* u = (float*)calloc(size, sizeof(float));
	float* data_3D = (float*)calloc(size_3D, sizeof(float));
	float* height_center = (float*)calloc(size, sizeof(float));
	float* height_x_edge = (float*)calloc(size_x, sizeof(float));
	float* height_y_edge = (float*)calloc(size_y, sizeof(float));
	float* data_edge_x = (float*)calloc(2*size_x, sizeof(float));
	float* data_edge_y = (float*)calloc(2*size_y, sizeof(float));


	char fileName[] = "brick_fines.txt";

	float *u_gpu, *data_3D_gpu, *data_edge_x_gpu, *data_edge_y_gpu;

  size_t memSize = size*sizeof(float);
  size_t memSize_3D = size_3D*sizeof(float);

	cudaMalloc( (void**)&u_gpu, memSize );
	cudaMalloc( (void**)&data_3D_gpu, memSize_3D );
	cudaMalloc( (void**)&data_edge_x_gpu, 2*size_x*sizeof(float) );
	cudaMalloc( (void**)&data_edge_y_gpu, 2*size_y*sizeof(float) );

	//init
	initialization(u, nx, ny, h, 3);
	read_txt(height_center, height_x_edge, height_y_edge, fileName, nx);
	init_surface_height_map(data_3D, height_center, nx, ny, h);
	init_height_map_edge(data_edge_x, data_edge_y, height_x_edge, height_y_edge, nx, ny, h);



	cudaMemcpy( u_gpu, u, memSize, cudaMemcpyHostToDevice );
	cudaMemcpy( data_3D_gpu, data_3D, memSize_3D, cudaMemcpyHostToDevice );
	cudaMemcpy( data_edge_x_gpu, data_edge_x, 2*size_x*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( data_edge_y_gpu, data_edge_y, 2*size_y*sizeof(float), cudaMemcpyHostToDevice );

	// dim3 grid, block;
  //
  // grid.x = GRID_SIZE;
  // grid.y = GRID_SIZE;
  // grid.z = 1;
  //
  // block.x = BLOCK_SIZE/GRID_SIZE;
  // block.y = BLOCK_SIZE/GRID_SIZE;
  // block.z = 1;
  // dim3 block(16,16);
  // dim3 grid;
  // grid.x = (nx + block.x - 1)/block.x;
  // grid.y = (ny + block.y - 1)/block.y;

  int Nblocks = (nx*nx + 255)/256;
  int Nthreads = 256;

	// PARAMETER
	 float tau = 0.001f ;
	 char title[50];
	// float e = 0.01;
	// float eta = 0.005;
	// float G = 5;
	// float beta = 0.0;
	int n_passe = 100;

  // plt::clf();
  //
  // sprintf(title, "Time = %f", 0.0);
  // const int colors = 1;
  //
  // plt::title(title);
  // plt::imshow(&(u[0]), ny, nx, colors);
  //
  // // Show plots
  // plt::pause(3);

  struct timeval start, end;
  gettimeofday(&start, NULL);


	//LOOP IN TIME
	for(int t = 0; t < 100; t++){
		for(int p=0; p<n_passe; p++){

			//Flux in direcion (di, dj) = (1,0) Horizontal
			// int di = 1;
			// int dj = 0;

			for(int rho=0; rho<4; rho++){
				//paralleliser

				flux_x<<<Nblocks, Nthreads>>>(u_gpu, data_3D_gpu, data_edge_x_gpu, 1, 0, rho);
        //grid, block
			}

			//Flux in direcion (di, dj) = (0,1) Vertical
			// di = 0;
			// dj = 1;

			for(int rho=0; rho<4; rho++){
				//paralleliser

				flux_y<<<Nblocks, Nthreads>>>(u_gpu, data_3D_gpu, data_edge_y_gpu, 0, 1, rho);

			}
		}
		// cudaDeviceSynchronize();
		// cudaError_t error = cudaGetLastError();
		// if (error != cudaSuccess) {
		//   fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
		// }

		cudaMemcpy( u, u_gpu, size*sizeof(float), cudaMemcpyDeviceToHost );

    // float max = 0;
    // int max_i, max_j;
    // int i,j;
    // for(int k=0; k<size; k++){
    //   i = k % nx;
    //   j = k / nx;
    //   if(u[nx*j+i]>max){
    //     max = u[nx*j+i];
    //     max_i = i;
    //     max_j = j;
    //   }
    // }
    // printf("Maximum = %f at i = %d, j = %d \n", max, max_i, max_j);

		// plt::clf();
    //
		// sprintf(title, "Time = %f", t*tau*n_passe/10.0f);
		// const int colors = 1;
    //
    // plt::title(title);
    // plt::imshow(&(u[0]), ny, nx, colors);
    //
    // // Show plots
    // plt::pause(0.0000001);

	}

  gettimeofday(&end, NULL);

  double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
         end.tv_usec - start.tv_usec) / 1.e6;
  printf("Time taken: %f \n", delta);


	//free memory
	free(u);
	free(data_3D);
	free(height_center);
	free(height_x_edge); free(height_y_edge);
	free(data_edge_x); free(data_edge_y); 

	cudaFree(u_gpu);
  cudaFree(data_3D_gpu);
  cudaFree(data_edge_x_gpu);
  cudaFree(data_edge_y_gpu);

	printf("\n *Happy computer sound* \n");

	return 0;
}


int parity(int di, int dj, int i, int j, int rho){
	return ((dj+1)*i + (di+1)*j + rho) % 4;
}
