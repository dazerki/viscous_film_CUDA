
#include "window.h"
#include "shaders.h"
#include "viscous.h"
#include "kernel.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <cuda.h>



#define GRID_SIZE 512
#define BLOCK_SIZE 16

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


	char fileName[] = "../src/brick_fines.txt";

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

  int Nblocks = (nx*nx + 255)/256;
  int Nthreads = 256;

  // Initialise window
  GLFWwindow *window = init_window();

  // Initialise shaders
  init_shaders();

  // Create Vertex Array Object
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // Create a Vertex Buffer Object for positions
  GLuint vbo_pos;
  glGenBuffers(1, &vbo_pos);

	GLfloat positions[2*nx*nx];
  for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nx; j++) {
          int ind = j*nx+i;
          positions[2*ind  ] = (float)(1.0 - 2.0*i/(nx-1));
          positions[2*ind+1] = (float)(1.0 - 2.0*j/(nx-1));
      }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

  // Specify vbo_pos' layout
  GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
  glEnableVertexAttribArray(posAttrib);
  glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

  // Create an Element Buffer Object and copy the element data to it
  GLuint ebo;
  glGenBuffers(1, &ebo);

	GLuint elements[4*(nx-1)*(nx-1)];
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < nx-1; j++) {
            int ind  = i*nx+j;
            int ind_ = i*(nx-1)+j;

            elements[4*ind_  ] = ind;
            elements[4*ind_+1] = ind+1;
            elements[4*ind_+2] = ind+nx;
            elements[4*ind_+3] = ind+nx+1;
        }
    }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

	// Create a Vertex Buffer Object for colors
  GLuint vbo_colors;
  glGenBuffers(1, &vbo_colors);

  GLfloat colors[nx*nx];
  for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nx; j++) {
          int ind = i*nx+j;
          colors[ind] = (float) u[ind];
      }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);

  // Specify vbo_color's layout
  GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
  glEnableVertexAttribArray(colAttrib);
  glVertexAttribPointer(colAttrib, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// PARAMETER
	float tau = 0.001f ;
	int n_passe = 100;

  // struct timeval start, end;
  // gettimeofday(&start, NULL);


	//LOOP IN TIME
  while(!glfwWindowShouldClose(window)) {
  	for(int p=0; p<n_passe; p++){
  		for(int rho=0; rho<4; rho++){
  			flux_x<<<Nblocks, Nthreads>>>(u_gpu, data_3D_gpu, data_edge_x_gpu, 1, 0, rho);
  		}

  		for(int rho=0; rho<4; rho++){
  			flux_y<<<Nblocks, Nthreads>>>(u_gpu, data_3D_gpu, data_edge_y_gpu, 0, 1, rho);
  		}
  	}

  	cudaMemcpy( u, u_gpu, size*sizeof(float), cudaMemcpyDeviceToHost );

    glfwSwapBuffers(window);
  	glfwPollEvents();

  	// Clear the screen to black
  	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  	glClear(GL_COLOR_BUFFER_BIT);

  	for (int i = 0; i < nx*nx; i++) {
  			colors[i] = (float) (u[i]);
  	}

  	glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  	glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);


  	// Draw elements
  	glDrawElements(GL_LINES_ADJACENCY, 4*(nx-1)*(nx-1), GL_UNSIGNED_INT, 0);

  	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
  			glfwSetWindowShouldClose(window, GL_TRUE);

  }

  // gettimeofday(&end, NULL);
  //
  // double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
  //        end.tv_usec - start.tv_usec) / 1.e6;
  // printf("Time taken: %f \n", delta);


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
