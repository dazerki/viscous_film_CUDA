#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_x(float *u, float *data_3D, float *data_edge, int di, int dj, int rho)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 1200;
	int ny = 1200;


	int i = (int) k % nx;
	int j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;

	if (rho_ij == 3){

		int i_p, j_p;
		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float H_p, H_q, T_p, T_q, ct_p, ct_q;
		float k_E, H_E;
		float u_p, u_q;
	  float h = 1.0f/nx;

	  float tau = 0.00025f ;
		float e = 0.01f;
		float eta = 0.00f;
		float G = 13.0f;
		float beta = 0.0f;
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

		H_p = data_3D[(nx*j_p + i_p)*3];
		H_q = data_3D[(nx*j + i)*3];

		T_p = data_3D[(nx*j_p + i_p)*3 + 1];
		T_q = data_3D[(nx*j + i)*3 + 1];

		ct_p = data_3D[(nx*j_p + i_p)*3 + 2];
		ct_q = data_3D[(nx*j + i)*3 + 2];

		k_E = data_edge[((nx+1)*j + i)*2];
		H_E = data_edge[((nx+1)*j + i)*2 + 1];

		W_q = G*(ny-j-0.5f)*h - H_q;
		W_p = G*(ny-j_p-0.5f)*h - H_p;

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0f)*(u_p*u_p + u_q*u_q);

		theta = h*h + (tau*M*(8.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
		f = -(M*h/(theta)) * ((5.0f*e + eta)*(u_q - u_p) - e*(lap_q - lap_p)*h + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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

		u[nx*j + i] = u_q + delta_u;
		u[nx*j_p + i_p] = u_p - delta_u;

	}
}

__global__ void flux_y(float *u, float *data_3D, float *data_edge, int di, int dj, int rho)
{
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 1200;
	int ny = 1200;


	int i = (int) k % nx;
	int j = (int) k / nx;


	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;
	if (rho_ij == 3){

		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float H_p, H_q, T_p, T_q, ct_p, ct_q;
		float k_E, H_E;
		int i_p, j_p;

		float u_p, u_q;
	  float h = 1.0f/nx;

	  float tau = 0.00025f ;
		float e = 0.01f;
		float eta = 0.00f;
		float G = 13.0f;
		float beta = 0.0f;
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

		H_p = data_3D[(nx*j_p + i_p)*3];
		H_q = data_3D[(nx*j + i)*3];

		T_p = data_3D[(nx*j_p + i_p)*3 + 1];
		T_q = data_3D[(nx*j + i)*3 + 1];

		ct_p = data_3D[(nx*j_p + i_p)*3 + 2];
		ct_q = data_3D[(nx*j + i)*3 + 2];

		k_E = data_edge[((nx)*j + i)*2];
		H_E = data_edge[((nx)*j + i)*2 + 1];

		W_q = G*(ny-j-0.5f)*h - H_q;

		if(j==0){
			W_p = G*(ny-(-1.0f)-0.5f)*h - H_p;
		}else{
			W_p = G*(ny-j_p-0.5f)*h - H_p;
		}

		M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0f)*(u_p*u_p + u_q*u_q);

		theta = h*h + (tau*M*(8.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
		f = -(M*h/(theta)) * ((5.0f*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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
		// if(delta_u<-0.05f){
		// 	printf("HERE, val = %f, f = %f \n", val, f);
		// }
		u[nx*j + i] = u_q + delta_u;
		u[nx*j_p + i_p] = u_p - delta_u;
  }
}

__global__ void flux(float *u, float* data_3D_gpu, float* data_edge_gpu, float *flx, int nx){
	//position in the grid
	int pos_x = 8*(blockIdx.x * blockDim.x + threadIdx.x) - 1;
	int pos_y = 8*(blockIdx.y * blockDim.y + threadIdx.y) - 1;

	//access memory
	float* u_local[100];
	for(int j=0; j<10; j++){
		for(int i=0; i<10; i++){
			u_local[10*j+i] = u[(pos_y+j)*nx + (pos_x+i)];
		}
	}

	float* data_3D_local[300];
	for(int j=0; j<10; j++){
		for(int i=0; i<10; i++){
			data_3D_local[(10*j+i)*3] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3];
			data_3D_local[(10*j+i)*3 + 1] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3 + 1];
			data_3D_local[(10*j+i)*3 + 2] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3 + 2];
		}
	}

	float* data_edge_local[200];
	for(int j=0; j<10; j++){
		for(int i=0; i<10; i++){
			data_edge_local[(10*j+i)*2] = data_edge_gpu[((pos_y+j)*nx + (pos_x+i))*2];
			data_edge_local[(10*j+i)*2 + 1] = data_edge_gpu[((pos_y+j)*nx + (pos_x+i))*2 + 1];
		}
	}

	float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
	float H_p, H_q, T_p, T_q, ct_p, ct_q;
	float k_E, H_E;
	int i_p, j_p;
	int di, dj

	float u_p, u_q;
	float h = 1.0f/nx;

	float tau = 0.00025f ;
	float e = 0.01f;
	float eta = 0.00f;
	float G = 13.0f;
	float beta = 0.0f;

	for(int direction=0, direction<2; direction++){
		if(direction==0){ //horizontal
			di = 1;
			dj = 0;
		} else { //vertical
			di = 0;
			dj = 1;
		}
		for(int i=1; i<9; i++){
			for(int j=1; j<9; j++){
				i_p = i - di;
				j_p = j - dj;

				lap_q = (u_local[10*j + (i+1)] + u_local[10*(j+1) + i] + u_local[10*(j) + i-1]);
				lap_p = (u_local[nx*j_p + (i_p-1)] + u_local[nx*(j_p) + i_p+1] + u_local[nx*(j_p-1) + i_p]);

				u_p = u_local[10*j_p + i_p];
				u_q = u_local[10*j + i];

				H_p = data_3D_local[(10*j_p + i_p)*3];
				H_q = data_3D_local[(10*j + i)*3];

				T_p = data_3D_local[(10*j_p + i_p)*3 + 1];
				T_q = data_3D_local[(10*j + i)*3 + 1];

				ct_p = data_3D_local[(10*j_p + i_p)*3 + 2];
				ct_q = data_3D_local[(10*j + i)*3 + 2];

				k_E = data_edge_local[(10*j + i)*2];
				H_E = data_edge_local[(10*j + i)*2 + 1];

				W_q = G*(ny-j-0.5f)*h - H_q;
				W_p = G*(ny-j_p-0.5f)*h - H_p;

				M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0f)*(u_p*u_p + u_q*u_q);

				theta = h*h + (tau*M*(8.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
				f = -(M*h/(theta)) * ((5.0f*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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

				flx[(pos_y+j)*nx + (pos_x+i)] += delta_u;
				flx[(pos_y+j_p)*nx + (pos_x+i_p)] -= delta_u;
			}
		}
	}

}
