#include <cuda.h>
#include <stdio.h>

#include "kernel.h"


__global__ void flux_block(float *u, float* data_3D_gpu, float* data_edge_gpu, float *flx, int nx){
	//position in the grid
	int k = (blockIdx.x * blockDim.x + threadIdx.x);
	int pos_x = 8*(8*k/nx) - 2;
	int pos_y = 8*(k%(nx/8)) - 2;


	//access memory
	float u_local[121];
	float data_3D_local[363];
	float data_edge_local[242];

	if(pos_x == -1 || pos_x == -2 || pos_x == nx-10 || pos_y == -1 || pos_y == -2 || pos_y == nx-10){

		//u local
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){
				u_local[11*j+i] = u[(pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx];
			}
		}


		//data_3D
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){
				data_3D_local[(11*j+i)*3] = data_3D_gpu[((pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx)*3];
				data_3D_local[(11*j+i)*3 + 1] = data_3D_gpu[((pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx)*3 + 1];
				data_3D_local[(11*j+i)*3 + 2] = data_3D_gpu[((pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx)*3 + 2];
			}
		}

		//data_edge
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){
				data_edge_local[(11*j+i)*2] = data_edge_gpu[((pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx)*2];
				data_edge_local[(11*j+i)*2 + 1] = data_edge_gpu[((pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx)*2 + 1];
			}
		}
	} else {
		//u local
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){

				u_local[11*j+i] = u[(pos_y+j)*nx + (pos_x+i)];
			}
		}

		//data_3D
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){
				if(((pos_y+j)*nx + (pos_x+i))*3 > 3*nx*nx){
					printf("index = %d \n", ((pos_y+j)*nx + (pos_x+i))*3);
				}
				data_3D_local[(11*j+i)*3] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3];
				data_3D_local[(11*j+i)*3 + 1] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3 + 1];
				data_3D_local[(11*j+i)*3 + 2] = data_3D_gpu[((pos_y+j)*nx + (pos_x+i))*3 + 2];
			}
		}

		//data_edge
		for(int j=0; j<11; j++){
			for(int i=0; i<11; i++){
				data_edge_local[(11*j+i)*2] = data_edge_gpu[((pos_y+j)*nx + (pos_x+i))*2];
				data_edge_local[(11*j+i)*2 + 1] = data_edge_gpu[((pos_y+j)*nx + (pos_x+i))*2 + 1];
			}
		}
	}






	float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
	float H_p, H_q, T_p, T_q, ct_p, ct_q;
	float k_E, H_E;
	int i_p, j_p;
	int di, dj;

	float u_p, u_q;
	float h = 1.0f/nx;

	float tau = 0.0001f ;
	float e = 0.01f;
	float eta = 0.00f;
	float G = 5.0f;
	float beta = 0.0f;

	for(int direction=0; direction<2; direction++){
		if(direction==0){ //horizontal
			di = 1;
			dj = 0;
		} else { //vertical
			di = 0;
			dj = 1;
		}
		for(int i=2; i<10; i++){
			for(int j=2; j<10; j++){
				i_p = i - di;
				j_p = j - dj;

				lap_q = (u_local[11*j + (i+1)] + u_local[11*(j+1) + i] + u_local[11*(j) + i-1]);
				lap_p = (u_local[11*j_p + (i_p-1)] + u_local[11*(j_p) + i_p+1] + u_local[11*(j_p-1) + i_p]);

				u_p = u_local[11*j_p + i_p];
				u_q = u_local[11*j + i];

				H_p = data_3D_local[(11*j_p + i_p)*3];
				H_q = data_3D_local[(11*j + i)*3];

				T_p = data_3D_local[(11*j_p + i_p)*3 + 1];
				T_q = data_3D_local[(11*j + i)*3 + 1];

				ct_p = data_3D_local[(11*j_p + i_p)*3 + 2];
				ct_q = data_3D_local[(11*j + i)*3 + 2];

				k_E = data_edge_local[(11*j + i)*2];
				H_E = data_edge_local[(11*j + i)*2 + 1];

				W_q = G*(nx-j-0.5f)*h - H_q;
				W_p = G*(nx-j_p-0.5f)*h - H_p;

				M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0f)*(u_p*u_p + u_q*u_q);

				theta = h*h + (tau*M*(4.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
				f = (M*h/(theta)) * (eta*(u_p - u_q) + (e/2.0f)*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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

				if(pos_x == -1 || pos_x == -2 || pos_x == nx-10 || pos_y == -1 || pos_y == -2 || pos_y == nx-10){
					flx[(pos_y+j+nx)%nx *nx + (pos_x+i+nx)%nx] += delta_u;
					flx[(pos_y+j_p+nx)%nx *nx + (pos_x+i_p+nx)%nx] -= delta_u;
				} else {
					flx[(pos_y+j)*nx + (pos_x+i)] += delta_u;
					flx[(pos_y+j_p)*nx + (pos_x+i_p)] -= delta_u;
				}


			}
		}
	}

}

__global__ void update_u(float *u, float* flux){
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	u[k] += flux[k];
}
