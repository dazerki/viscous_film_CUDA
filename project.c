#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../matplotlib-cpp-master/matplotlibcpp.h"
#include <omp.h>
#include "viscous.h"



namespace plt = matplotlibcpp;

int parity(int di, int dj, int i, int j, int rho);
float min(float a, double b);
float max(float a, double b);

int main(int argc, char *argv[]){

	int nx = 512;
	int ny = 512;
	double h = 1.0/nx ;

	// memory allocation
	float* u = (float*)calloc(nx*ny, sizeof(float));
	double* H = (double*)calloc(nx*ny, sizeof(double));
	double* T = (double*)calloc(nx*ny, sizeof(double));
	double* ctheta = (double*)calloc(nx*ny, sizeof(double));
	double* height_center = (double*)calloc(nx*ny, sizeof(double));
	double* height_x_edge = (double*)calloc((nx+1)*ny, sizeof(double));
	double* height_y_edge = (double*)calloc(nx*(ny+1), sizeof(double));
	double* H_edge_x = (double*)calloc((nx+1)*ny, sizeof(double));
	double* H_edge_y = (double*)calloc(nx*(ny+1), sizeof(double));
	double* k_x = (double*)calloc((nx+1)*ny, sizeof(double));
	double* k_y = (double*)calloc(nx*(ny+1), sizeof(double));
	char fileName[] = "brick_fines.txt";

	//init
	initialization(u, nx, ny, h, 3);
	read_txt(height_center, height_x_edge, height_y_edge, fileName, nx);
	init_surface_height_map(H, T, ctheta, height_center, nx, ny, h);
	init_height_map_edge(H_edge_x, H_edge_y, k_x, k_y, height_x_edge, height_y_edge, nx, ny, h);

	//BORDER
	// for(int i=0; i<nx; i++){
	// 	u[i] = 0.;
	// 	u[nx*(ny-1) + i] = 0.;
	// }
	//
	// for(int j=0; j<ny; j++){
	// 	u[nx*j] = 0.;
	// 	u[nx*j + nx-1] = 0.;
	// }

	// PARAMETER
	double tau = 0.001 ;
	double e = 0.01;
	double eta = 0.005;
	double G = 5;
	double sigma = 0.075;
	double beta = 0.0;
	int n_passe = 100;
	char title[50];
	float u_tot;

 	omp_set_num_threads(6);

	double start, end;
	start = omp_get_wtime();
	//LOOP IN TIME
	for(int t = 0; t < 500; t++){
		for(int p=0; p<n_passe; p++){

			// u_tot = 0.0;
			// for(int i = 0; i <nx*ny; i++){
			// 	u_tot = u_tot + u[i];
			// }
			// printf("u_tot = %f \n", u_tot);

			//Flux in direcion (di, dj) = (1,0) Horizontal
			int di = 1;
			int dj = 0;

			for(int rho=0; rho<4; rho++){
				 #pragma omp parallel for
				for(int k=0; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					double W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					double H_p, H_q, T_p, T_q, ct_p, ct_q;
					double k_E, H_E;
					int i,j;
					double mini;
					float u_p, u_q;


					i = (int) k % nx;
					j = (int) k / nx;

					rho_ij = parity(di, dj, i, j, rho);

					if (rho_ij == 3){
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

						H_p = H[nx*j_p + i_p];
						H_q = H[nx*j + i];

						T_p = T[nx*j_p + i_p];
						T_q = T[nx*j + i];

						ct_p = ctheta[nx*j_p + i_p];
						ct_q = ctheta[nx*j + i];

						k_E = k_x[(nx+1)*j + i];
						H_E = H_edge_x[(nx+1)*j + i];

						// i_p = (i - di + nx)%nx;
						// j_p = (j - dj + ny)%ny;
						//
						// lap_q = (u[nx*((ny+j)%ny) + (nx+(i+1))%nx] + u[nx*((ny+j+1)%ny) + (nx+i)%nx] + u[nx*((ny+(j-1))%ny) + (i+nx)%nx]);
						// lap_p = (u[nx*((j_p+ny)%ny) + ((i_p-1+nx)%nx)] + u[nx*((j_p+1+ny)%ny) + (i_p+nx)%nx] + u[nx*((j_p-1+ny)%ny) + (i_p+nx)%nx]);
						//
						// u_p = u[nx*((ny+j_p)%ny) + (nx+i_p)%nx];
						// u_q = u[nx*((ny+j)%ny) + (nx+i)%nx];



						W_q = G*(ny-j-0.5)*h - H[nx*j+i];
						W_p = G*(ny-j_p-0.5)*h - H[nx*j_p+i_p];

						//M = (2.0/3.0) * 1.0/(1.0/(u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j_p + i_p]) + 1.0/(u[nx*j + i]*u[nx*j + i]*u[nx*j + i]));
						M = 2.0 * u_p*u_p * u_q*u_q /(3.0*(u_q + u_p)) + (e/6.0)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0)*(u_p*u_p + u_q*u_q);

						//3D
						theta = h*h + (tau*M*(8.0*e + 2.0*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
						f = -(M*h/(theta)) * ((5.0*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

						// nouveau
						// theta = h*h + (2.0*tau*M*(4.0*e + eta));
						// f = -(M*h/(theta)) * ((5.0*e+eta)*(u_q - u_p) - e*(lap_q - lap_p)); //+ W_q-W_p

						//auteurs
						// theta = 1.0 + (2.0*tau*M*(5.0*e + eta));
						// f = -(M/(theta)) * ((5.0*e+eta)*(u_q - u_p) - e*(lap_q - lap_p)); //+ W_q-W_p


						// mini = min(u_p, tau*f/h);
						// delta_u = max(-u_q, mini);
						double val = tau*f/h;
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

						// if(delta_u > 0.1){
						// 	printf("delta u = %f at i=%d and j=%d \n", delta_u, i, j);
						// }

						u[nx*j + i] = u_q + delta_u;
						u[nx*j_p + i_p] = u_p - delta_u;


					}
				}
			}

			//Flux in direcion (di, dj) = (0,1) Vertical
			di = 0;
			dj = 1;

			for(int rho=0; rho<4; rho++){
				#pragma omp parallel for
				for(int k=0; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					double W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					double H_p, H_q, T_p, T_q, ct_p, ct_q;
					double k_E, H_E;
					int i,j;
					double mini;
					float u_p, u_q;

					i = (int) k % nx;
					j = (int) k / nx;

					rho_ij = parity(di, dj, i, j, rho);
					if (rho_ij == 3){
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

						H_p = H[nx*j_p + i_p];
						H_q = H[nx*j + i];

						T_p = T[nx*j_p + i_p];
						T_q = T[nx*j + i];

						ct_p = ctheta[nx*j_p + i_p];
						ct_q = ctheta[nx*j + i];

						k_E = k_y[(nx)*j + i];
						H_E = H_edge_y[(nx)*j + i];


						// i_p = (i - di + nx)%nx;
						// j_p = (j - dj + ny)%ny;
						//
						// lap_q = (u[nx*((j+ny)%ny) + (i+1+nx)%nx] + u[nx*((j+1+ny)%ny) + (i+nx)%nx] + u[nx*((j+ny)%ny) + (i-1+nx)%nx]);
						// lap_p = (u[nx*((j_p+ny)%ny) + (i_p-1+nx)%nx] + u[nx*((j_p+ny)%ny) + (i_p+1+nx)%nx] + u[nx*((j_p-1+ny)%ny) + (i_p+nx)%nx]);
						//
						// u_p = u[nx*((ny+j_p)%ny) + (i_p+nx)%nx];
						// u_q = u[nx*((j+ny)%ny) + (i+nx)%nx];

						//nouveau
						W_q = G*(ny-j-0.5)*h - H[nx*j+i];

						//printf("gravité: %f, surface: %f \n",G*(ny-j-0.5)*h,H[nx*j+i]);
						if(j==0){
							W_p = G*(ny-(-1)-0.5)*h - H[nx*(ny-1) + i_p];
						}else{
							W_p = G*(ny-j_p-0.5)*h - H[nx*j_p+i_p];
						}

						//auteurs
						// W_q = G*(ny-j-0.5);
						// if(j==0){
						// 	W_p = G*(ny-(-1)-0.5);
						// }else{
						// 	W_p = G*(ny-j_p-0.5);
						// }


						//M = (2.0/3.0) * 1.0/(1.0/(u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j_p + i_p]) + 1.0/(u[nx*j + i]*u[nx*j + i]*u[nx*j + i]));
						M = 2.0 * u_q*u_q * u_p*u_p /(3.0*(u_q + u_p)) + (e/6.0)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0)*(u_p*u_p + u_q*u_q);

						//nouveau
						// theta = h*h + (2.0*tau*M*(4.0*e + eta));
						// f = -(M*h/(theta)) * ((5.0*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p);

						//3D
						theta = h*h + (tau*M*(8.0*e + 2.0*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
						f = -(M*h/(theta)) * ((5.0*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

						//auteurs
						// theta = 1.0 + (2.0*tau*M*(5.0*e + eta));
						// f = -(M/(theta)) * ((5.0*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p);



						// mini = min(u_p,tau*f/h);
						// delta_u = max(-u_q, mini);

						double val = tau*f/h;
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

						// if (i==256 && j==300){
						// 	printf("Gravity : %f, viscous: %f, stab: %f and delta_u = %f \n ", W_q-W_p, 5.0*e*(u_q-u_p)-e*(lap_q-lap_p), eta*(u_q-u_p), delta_u);
						// }

						u[nx*j + i] = u[nx*j + i] + delta_u;
						u[nx*j_p + i_p] = u[nx*j_p + i_p] - delta_u;
				}
			}
		}
	}

		plt::clf();

		sprintf(title, "Time = %f", t*tau*n_passe/10.0);
		const int colors = 1;

    plt::title(title);
    plt::imshow(&(u[0]), ny, nx, colors);

    // Show plots
    plt::pause(0.1);

		//printf("t = %d\n", t);

	}
		end = omp_get_wtime();
		printf("time taken: %f seconds", end-start);

	//free memory
	free(u);

	printf("\n *Happy computer sound* \n");

	return 0;
}


int parity(int di, int dj, int i, int j, int rho){
	return ((dj+1)*i + (di+1)*j + rho) % 4;
}

float min(float a, double b){
	if(a<b){
		return a;
	} else{
		return b;
	}
}

float max(float a, double b){
	if(a>b){
		return a;
	} else{
		return b;
	}
}
