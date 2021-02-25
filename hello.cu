// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <stdio.h>

const int N = 16;
const int blocksize = 16;

__global__
void flux(float *u, float *H, float *T, float *ctheta, float *k_edge, float *H_edge, float di, float dj)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	double tau = 0.001 ;
	double e = 0.01;
	double eta = 0.005;
	double G = 5;
	double beta = 0.0;
	
	int rho_ij, i_p, j_p;
	double W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
	double H_p, H_q, T_p, T_q, ct_p, ct_q;
	double k_E, H_E;
	int i,j;
	double mini;
	float u_p, u_q;

	i = (int) k % nx;
	j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;
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

		if(di==1){
			k_E = k_y[(nx)*j + i];
			H_E = H_edge_y[(nx)*j + i];
		} else{
			k_E = k_y[(nx)*j + i];
			H_E = H_edge_y[(nx)*j + i];
		}

		//nouveau
		W_q = G*(ny-j-0.5)*h - H_q;
		if(di==1){
			W_p = G*(ny-j_p-0.5)*h - H[nx*j_p+i_p];
		} else {
			if(j==0){
				W_p = G*(ny-(-1)-0.5)*h - H_p;
			}else{
				W_p = G*(ny-j_p-0.5)*h - H_p;
			}
		}


		M = 2.0 * u_q*u_q * u_p*u_p /(3.0*(u_q + u_p)) + (e/6.0)*u_q*u_q*u_p*u_p*(H_E+k_E) + (beta/2.0)*(u_p*u_p + u_q*u_q);

		//3D
		theta = h*h + (tau*M*(8.0*e + 2.0*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
		f = -(M*h/(theta)) * ((5.0*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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

		u[nx*j + i] = u[nx*j + i] + delta_u;
		u[nx*j_p + i_p] = u[nx*j_p + i_p] - delta_u;
}


int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );
	cudaFree( bd );

	printf("%s\n", a);
	return EXIT_SUCCESS;
}
