#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_block(float *u, float* data_3D_gpu, float* data_edge_gpu, float* flx_x, float* flx_y, int nx){
	//position in the grid
	int pos_x = 4*(blockIdx.x * blockDim.x + threadIdx.x)-2;
	int pos_y = 4*(blockIdx.y * blockDim.y + threadIdx.y)-2;

	int pos_data3D_x = 4*(blockIdx.x * blockDim.x + threadIdx.x)-1;
	int pos_data3D_y = 4*(blockIdx.y * blockDim.y + threadIdx.y)-1;



	int pos_x_data = 4*(blockIdx.x * blockDim.x + threadIdx.x);
	int pos_y_data = 4*(blockIdx.y * blockDim.y + threadIdx.y);

	int pos_block_i = threadIdx.x * 4 + 2;
	int pos_block_j = threadIdx.y * 4 + 2;

	int pos_data3D_i = 4* threadIdx.x+1;
	int pos_data3D_j = 4*threadIdx.y+1;

	int pos_block_x = threadIdx.x * 4;
	int pos_block_y = threadIdx.y * 4;

	int size_u_line = (4*blockDim.x+3);
	int size_data_line = 4*blockDim.x+1;
	int size_edge_line = 4*blockDim.x;

	//access memory
	__shared__ float u_local[1225]; //4*dimX+3) * (4*dimY+3
	__shared__ float data_3D_local[3267]; // 3*(4*dimX+1)*(dimY*4+1)
	__shared__ float data_edge_local[2178]; // 2*(4*dimX+1)*(dimY*4+1)
	//u local
	if(threadIdx.x == 0){
		for(int j=1; j<7; j++){
			for(int i=0; i<2; i++){
				u_local[size_u_line*(pos_block_y+j) + (i+pos_block_x)] = u[((pos_y+j+nx)%nx)*nx + ((pos_x+i+nx)%nx)];
			}
		}
	}
	if (threadIdx.x == blockDim.x-1){
		for(int j=2; j<6; j++){
			u_local[size_u_line*(pos_block_y+j) + (6+pos_block_x)] = u[((pos_y+j+nx)%nx)*nx + ((pos_x+6+nx)%nx)];
		}
	}
	if(threadIdx.y == 0){
		for(int j=0; j<2; j++){
			for(int i=2; i<6; i++){
				u_local[size_u_line*(pos_block_y+j) + (i+pos_block_x)] = u[((pos_y+j+nx)%nx)*nx + ((pos_x+i+nx)%nx)];
			}
		}

	}
	if (threadIdx.y == blockDim.y-1){
		for(int i=2; i<6; i++){
			u_local[size_u_line*(pos_block_y+6) + (i+pos_block_x)] = u[((pos_y+6+nx)%nx)*nx + ((pos_x+i+nx)%nx)];
		}
	}


	for(int j=2; j<6; j++){
		for(int i=2; i<6; i++){
			u_local[size_u_line*(pos_block_y+j) + (i+pos_block_x)] = u[(pos_y+j)*nx + (pos_x+i)];
		}
	}

	//data_3D
	if(threadIdx.x == 0){
		for(int j=1; j<5; j++){
			data_3D_local[(size_data_line*(pos_block_y+j) + (pos_block_x))*3] = data_3D_gpu[(((pos_data3D_y+j+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3];
			data_3D_local[(size_data_line*(pos_block_y+j) + (pos_block_x))*3 + 1] = data_3D_gpu[(((pos_data3D_y+j+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3 + 1];
			data_3D_local[(size_data_line*(pos_block_y+j) + (pos_block_x))*3 + 2] = data_3D_gpu[(((pos_data3D_y+j+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3 + 2];
		}

	}
	if(threadIdx.y == 0){
		for(int i=1; i<5; i++){
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + i))*3] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + i+nx)%nx))*3];
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + i))*3 + 1] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + i+nx)%nx))*3 + 1];
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + i))*3 + 2] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + i+nx)%nx))*3 + 2];
		}

	}

	for(int j=1; j<5; j++){
		for(int i=1; i<5; i++){
			data_3D_local[(size_data_line*(pos_block_y+j) + (i+pos_block_x))*3] = data_3D_gpu[((pos_data3D_y+j)*nx + (pos_data3D_x+i))*3];
			data_3D_local[(size_data_line*(pos_block_y+j) + (i+pos_block_x))*3 + 1] = data_3D_gpu[((pos_data3D_y+j)*nx + (pos_data3D_x+i))*3 + 1];
			data_3D_local[(size_data_line*(pos_block_y+j) + (i+pos_block_x))*3 + 2] = data_3D_gpu[((pos_data3D_y+j)*nx + (pos_data3D_x+i))*3 + 2];
		}
	}

	//data_edge
	for(int j=0; j<4; j++){
		for(int i=0; i<4; i++){
			data_edge_local[(size_edge_line*(pos_block_y+j) + (i+pos_block_x))*2] = data_edge_gpu[((pos_y_data+j)*nx + (pos_x_data+i))*2];
			data_edge_local[(size_edge_line*(pos_block_y+j) + (i+pos_block_x))*2 + 1] = data_edge_gpu[((pos_y_data+j)*nx + (pos_x_data+i))*2 + 1];
		}
	}
	__syncthreads();

	float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
	float H_p, H_q, T_p, T_q, ct_p, ct_q;
	float k_E, H_E;
	int i_p, j_p;
	int di, dj;

	float u_p, u_q;
	float h = 1.0f/nx;

	float tau = 0.0002f ;
	float e = 0.01f;
	float eta = 0.005f;
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
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				i_p = i - di;
				j_p = j - dj;

				if(direction == 0){
					lap_q = (u_local[size_u_line*(pos_block_j+j) + (i+pos_block_i+1)] + u_local[size_u_line*(pos_block_j+j+1) + (i+pos_block_i)] + u_local[size_u_line*(pos_block_j+j-1) + (i+pos_block_i)]);
					lap_p = (u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i-1)] + u_local[size_u_line*(pos_block_j+j_p+1) + (i_p+pos_block_i)] + u_local[size_u_line*(pos_block_j+j_p-1) + (i_p+pos_block_i)]);
					// if(lap_p-lap_q>0.1 || lap_q - lap_p >0.1){
					// 	printf("HERE (i,j) = (%d,%d), u_i-1 = %f, u_j+1 = %f, u_j-1 = %f\n", i_p+pos_block_i,j_p+pos_block_j, u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i-1)], u_local[size_u_line*(pos_block_j+j_p+1) + (i_p+pos_block_i)], u_local[size_u_line*(pos_block_j+j_p-1) + (i_p+pos_block_i)]);
					// }
				} else {
					lap_q = (u_local[size_u_line*(pos_block_j+j) + (i+pos_block_i+1)] + u_local[size_u_line*(pos_block_j+j+1) + (i+pos_block_i)] + u_local[size_u_line*(pos_block_j+j) + (i+pos_block_i-1)]);
					lap_p = (u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i-1)] + u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i+1)] + u_local[size_u_line*(pos_block_j+j_p-1) + (i_p+pos_block_i)]);
				}



				u_p = u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i)];
				u_q = u_local[size_u_line*(pos_block_j+j) + (i+pos_block_i)];

				H_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3];
				H_q = data_3D_local[(size_data_line*(pos_data3D_j+j) + (i+pos_data3D_i))*3];

				T_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3 + 1];
				T_q = data_3D_local[(size_data_line*(pos_data3D_j+j) + (i+pos_data3D_i))*3 + 1];

				ct_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3 + 2];
				ct_q = data_3D_local[(size_data_line*(pos_data3D_j+j) + (i+pos_data3D_i))*3 + 2];

				k_E = data_edge_local[(size_edge_line*(pos_block_y+j) + (i+pos_block_x))*2];
				H_E = data_edge_local[(size_edge_line*(pos_block_y+j) + (i+pos_block_x))*2 + 1];

				W_q = G*(nx-(pos_y_data+j)-0.5f)*h - H_q;
				W_p = G*(nx-(pos_y_data+j_p)-0.5f)*h - H_p;




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

				if(direction==0){
						flx_x[(pos_y_data+j)*nx + (pos_x_data+i)] = delta_u;
				} else {
						flx_y[(pos_y_data+j)*nx + (pos_x_data+i)] = delta_u;
				}

			}
		}
	}

}

__global__ void update_u(float *u, float* flux, int dir, int flag, int nx){
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = k%nx;
	int j = k/nx;
	int i_p, j_p;

	if(dir==0){ //horizontal
		i_p = i-1;
		j_p = j;
	} else { //vertical
		i_p = i;
		j_p = j-1;
	}
	if((i+(nx+1)*j)%2 == flag){
		u[k] += flux[k];
		u[nx*((j_p+nx)%nx)+((i_p+nx)%nx)] -= flux[k];
		flux[k] = 0.0f;
	}



}
