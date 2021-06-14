#include <cuda.h>
#include <stdio.h>


#include "kernel.h"

__global__ void flux_x(float *u, int rho)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 512;
	int ny = 512;
	int di = 1;
	int dj = 0;


	int i = (int) k % nx;
	int j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;

	if (rho_ij == 3){

		int i_p, j_p;
		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float u_p, u_q;
	  float h = 1.0f/nx;

	  float tau = 0.001f ;
		float e = 0.01f;
		float eta = 0.00f;
		float G = 13.0f;
		if (i==0){
			i_p = nx - 1;
			j_p = j - dj;
		} else{
			i_p = i - di;
			j_p = j - dj;
		}

		if (i==nx-1){
			if(j==0){
				lap_q = (u[nx*j] + u[nx*(j+1) + i] + u[nx*(ny-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(ny-1) + i_p]);
			} else if(j==ny-1){
				lap_q = (u[nx*j] + u [i] + u[nx*(j-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[i_p] + u[nx*(j_p-1) + i_p]);
			}
			else{
				lap_q = (u[nx*j + (0)] + u[nx*(j+1) + i] + u[nx*(j-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(j_p-1) + i_p]);
			}
		} else if(i==1){
			if(j==0){
				lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(ny-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(ny-1) + i_p]);
			} else if(j==ny-1){
				lap_q = (u[nx*j + (i+1)] + u[nx*(0) + i] + u[nx*(j-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(0) + i_p] + u[nx*(j_p-1) + i_p]);
			}
			else{
				lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(j_p-1) + i_p]);
			}
		} else if (j==0){
				lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(ny-1) + i]);
				lap_p = (u[nx*j_p + (nx-1)] + u[nx*(j_p+1) + i_p] + u[nx*(ny-1) + i_p]);
		} else if (j==ny-1){
				lap_q = (u[nx*j + (i+1)] + u [i] + u[nx*(j-1) + i]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[i_p] + u[nx*(j_p-1) + i_p]);
		} else{
			lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j-1) + i]);
			lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(j_p-1) + i_p]);
		}


		u_p = u[nx*j_p + i_p];
		u_q = u[nx*j + i];

		// W_q = G*(ny-j-0.5f)*h;
		// W_p = G*(ny-j_p-0.5f)*h;

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p));

		theta = h*h + (2.0f*tau*M*(5.0f*e + eta));
		f = (M*h/(theta)) * ((eta+5.0f*e)*(u_p - u_q) + (e)*(lap_q - lap_p));

		float val = tau*f/h;
		if(u_p<val){
			if(u_p > -u_q){
				delta_u = u_p;
			} else {
				delta_u = -u_q;
			}
		} else{
			if(val > -u_q){
				delta_u = val;
			} else {
				delta_u = -u_q;
			}
		}

		if(i == 0){
			u[nx*(j+nx)%nx + (i+nx)%nx] += delta_u;
			u[nx*(j_p+nx)%nx + (i_p+nx)%nx] -= delta_u;
		} else {
			u[nx*j + i] += delta_u;
			u[nx*j_p + i_p] -= delta_u;
		}


	}
}

__global__ void flux_y(float *u, int rho)
{
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 512;
	int ny = 512;
	int di = 0;
	int dj = 1;


	int i = (int) k % nx;
	int j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;
	if (rho_ij == 3){

		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		int i_p, j_p;

		float u_p, u_q;
	  float h = 1.0f/nx;

	  float tau = 0.001f ;
		float e = 0.01f;
		float eta = 0.00f;
		float G = 13.0f;

		if (j==0){
			i_p = i - di;
			j_p = ny - 1;
		} else {
			i_p = i - di;
			j_p = j - dj;
		}


		if (j==ny-1){
			if(i==0){
				lap_q = (u[nx*j + (i+1)] + u[nx*(0) + i] + u[nx*(j) + nx-1]);
				lap_p = (u[nx*j_p + (nx-1)] + u[nx*(j_p) + i_p+1] + u[nx*(j_p-1) + i_p]);
			} else if (i==nx-1){
				lap_q = (u[nx*j + (0)] + u[nx*(0) + i] + u[nx*(j) + i-1]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p) + 0] + u[nx*(j_p-1) + i_p]);
			} else {
				lap_q = (u[nx*j + (i+1)] + u[nx*(0) + i] + u[nx*(j) + i-1]);
				lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p) + i_p+1] + u[nx*(j_p-1) + i_p]);
			}
		} else if (j==1){
			if(i==0){
				lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j) + nx-1]);
				lap_p = (u[nx*j_p + (nx-1)] + u[nx*(j_p) + i_p+1] + u[nx*(ny-1) + i_p]);
			} else if(i==nx-1){
				lap_q = (u[nx*j + (0)] + u[nx*(j+1) + i] + u[nx*(j) + i-1]);
				lap_p = (u[nx*j_p + (i-1)] + u[nx*(j_p) + 0] + u[nx*(ny-1) + i_p]);
			} else {
				lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j) + i-1]);
				lap_p = (u[nx*j_p + (i-1)] + u[nx*(j_p) + i_p+1] + u[nx*(ny-1) + i_p]);
			}
		} else if (i==0){
			lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j) + nx-1]);
			lap_p = (u[nx*j_p + (nx-1)] + u[nx*(j_p) + i_p+1] + u[nx*(j_p-1) + i_p]);
		} else if (i==nx-1){
			lap_q = (u[nx*j + (0)] + u[nx*(j+1) + i] + u[nx*(j) + i-1]);
			lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p) + 0] + u[nx*(j_p-1) + i_p]);
		} else{
			lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j) + i-1]);
			lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p) + i_p+1] + u[nx*(j_p-1) + i_p]);
		}

		u_p = u[nx*j_p + i_p];
		u_q = u[nx*j + i];

		W_q = G*(ny-j-0.5f)*h;

		if(j==0){
			W_p = G*(ny-(-1.0f)-0.5f)*h;
		}else{
			W_p = G*(ny-j_p-0.5f)*h;
		}

		M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p));

		theta = h*h + (2.0f*tau*M*(5.0f*e + eta));
		f = (M*h/(theta)) * ((eta+5.0f*e)*(u_p - u_q) + (e)*(lap_q - lap_p) + W_p-W_q);

		float val = tau*f/h;
		if(u_p<val){
			if(u_p > -u_q){
				delta_u = u_p;
			} else {
				delta_u = -u_q;
			}
		} else{
			if(val > -u_q){
				delta_u = val;
			} else {
				delta_u = -u_q;
			}
		}

		if(j == 0){
			u[nx*((j+nx)%nx) + ((i+nx)%nx)] += delta_u;
			u[nx*((j_p+nx)%nx) + ((i_p+nx)%nx)] -= delta_u;
		} else {
			u[nx*j + i] += delta_u;
			u[nx*j_p + i_p] -= delta_u;
		}
  }
}
