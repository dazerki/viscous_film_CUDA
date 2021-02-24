#include "viscous.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initialization(float* u, int nx, int ny,  float h, int choice){

  for(int i=0; i<nx*ny; i++){
    u[i] = 0.04; // 0.0001
	}

  // GAUSSIANS
  if(choice == 1){

    gaussians(u, nx, ny, h);

  }
  // Center circle + line
  else if(choice == 2) {
    circle(u, nx, ny, 0.04);
  }

  else if(choice == 3) {
    simple_gaussian(u, nx, ny, h);
  }

}

void gaussians(float* u, int nx, int ny, float h){

  double mu_x[5] = {0.19, 0.2, 0.56, 0.6, 0.9};
	double mu_y[5] = {0.8, 0.45, 0.7, 0.3, 0.5};
	double sigma_x[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
	double sigma_y[5] = {0.07, 0.07, 0.07, 0.07, 0.07};
	double  x, y;
  float density;
	int i, j;
	float max = 0.0;


	for(int l=0; l<5; l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (float)(1.0/(100.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
  printf("Max = %f \n", max);

}

void circle(float* u, int nx, int ny, float value){
  for(int i=0; i<nx*ny; i++){
    if((i/nx - 256)*(i/nx - 256) + (i%nx -256)*(i%nx - 256) < 75*75){
    	u[i] = value;
    }
  }
}

void double_circle(float* u, int nx, int ny, float value){
  for(int i=0; i<nx*ny; i++){
    if((((i/nx - 150)*(i/nx - 150) + (i%nx -170)*(i%nx - 170) < 75*75) && i/nx > 175) || (((i/nx - 150)*(i/nx - 150) + (i%nx -342)*(i%nx - 342) < 75*75) && i/nx > 175)){
			u[i] = value;
		}
  }
}

void simple_gaussian(float* u, int nx, int ny, float h){
  double mu_x[1] = {0.5};
	double mu_y[1] = {0.15};
	double sigma_x[1] = {0.1};
	double sigma_y[1] = {0.07};
	double density, x, y;
	int i, j;
	double max = 0;

	for(int l=0; l<sizeof(mu_x); l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (1.0/(50.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
	printf("Max = %f \n", max);
}

void big_line(float* u, int nx, int ny, float value){
  for(int index=350*nx ; index<390*nx ; index++){
		u[index] = value;
	}
}

void merging_gaussian(float* u, int nx, int ny, float h){
  double mu_x[2] = {0.5, 0.65};
	double mu_y[2] = {0.6, 0.8};
	double sigma_x[2] = {0.1, 0.1};
	double sigma_y[2] = {0.1, 0.1};
	double density, x, y;
	int i, j;
	double max = 0;

	for(int l=0; l<sizeof(mu_x); l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (1.0/(100.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
	printf("Max = %f \n", max);

}

void init_surface(double* H, double* T, double* ctheta, int nx, int ny, double h){
  double val_H = 0; double val_T = 0; double val_ctheta = 0;
  double y = 0;
  for (int j=1; j<ny; j++){
    y = j*h;
    val_H = (1300.0*pow(y,3.0)*(pow(y,2.0)/(676.0*(100.0*pow(abs(y),4.0) - 100.0*y*pow(abs(y),2.0) + 26.0*pow(y,2.0))*(100.0*pow(y,2.0) - 100.0*y + 26.0)) + 1.0)*(y - 2.0*pow(abs(y),2.0))*(50.0*pow(y,2.0) - 50.0*y + 13.0))
          /(pow((1.0/(676.0*pow(abs(pow((10.0*y - 5.0),2.0) + 1.0),2.0)) + 1.0),(1.0/2.0))*(50.0*pow(abs(y),4.0) - 50.0*y*pow(abs(y),2.0) + 13.0*pow(y,2.0))*(1757600.0*pow(abs(y),4.0) - 6760000.0*y*pow(abs(y),4.0) - 1757600.0*y*pow(abs(y),2.0)
          + 6760000.0*pow(y,2.0)*pow(abs(y),2.0) - 6760000.0*pow(y,3.0)*pow(abs(y),2.0) + 6760000.0*pow(y,2.0)*pow(abs(y),4.0) + 456977.0*pow(y,2.0) - 1757600.0*pow(y,3.0) + 1757600.0*pow(y,4.0)));

    val_T = (1690000.0*pow(y,6.0)*pow((pow(y,2.0)/(676.0*(100.0*pow(abs(y),4.0) - 100.0*y*pow(abs(y),2.0) + 26.0*pow(y,2.0))*(100.0*pow(y,2.0) - 100.0*y + 26.0)) + 1.0),2.0)*pow((y - 2.0*pow(abs(y),2.0)),2.0)*pow((50.0*pow(y,2.0) - 50.0*y + 13.0),2.0))
            /((1.0/(676.0*pow(abs(pow((10.0*y - 5.0),2.0) + 1.0),2.0)) + 1.0)*pow((13.0*pow(y,2.0) - 50.0*y*pow(abs(y),2.0) + 50.0*pow(abs(y),4.0)),2.0)*pow(1757600.0*pow(y,4.0) - 6760000.0*pow(y,3.0)*pow(abs(y),2.0)
            - 1757600.0*pow(y,3.0) + 6760000.0*pow(y,2.0)*pow(abs(y),4.0) + 6760000.0*pow(y,2.0)*pow(abs(y),2.0) + 456977.0*pow(y,2.0) - 6760000.0*y*pow(abs(y),4.0) - 1757600.0*y*pow(abs(y),2.0) + 1757600.0*pow(abs(y),4.0),2.0));

    val_ctheta = -1.0/(26.0*pow(1.0/(676.0*pow(abs(pow(10.0*y - 5.0,2.0) + 1.0),2.0)) + 1.0,(1.0/2.0))*(25.0*pow(2.0*y - 1.0,2.0) + 1.0));

    for(int i=0; i<nx; i++){
      H[nx*j + i] = val_H;
      T[nx*j + i] = val_T;
      ctheta[nx*j + i] = val_ctheta;
    }
  }
  for(int i=0; i<nx; i++){
    H[i] = 0.00285;
    T[i] = 0.000008;
    ctheta[i] = -0.001482;
  }
}

double x_deriv(double h_r, double h_l, double dx){
  return (h_r-h_l)/(2.0*dx);
}

double y_deriv(double h_u, double h_d, double dy){
  return (h_u-h_d)/(2.0*dy);
}

double xx_deriv(double h_r, double h_l, double h, double dx){
  return (h_r - 2.0*h + h_l)/(dx*dx);
}

double yy_deriv(double h_u, double h_d, double h, double dy){
  return (h_u - 2.0*h + h_d)/(dy*dy);
}

double xy_deriv(double h_ur, double h_dr, double h_ul, double h_dl, double dx, double dy){
  return (h_ur - h_dr - h_ul + h_dl)/(4.0*dx*dy);
}

void init_surface_height_map(double* H, double* T, double* ctheta, double* height, int nx, int ny, double dx){
  double h_ul, h_u, h_ur, h_l, h, h_r, h_dl, h_d, h_dr;
  double h_x, h_xx, h_y, h_yy, h_xy;
  double K;
  for (int j=1; j<ny-1; j++){
    for(int i=1; i<nx-1; i++){
      h = height[nx*j+i];
      h_ul = height[nx*(j+1) + i-1];
      h_l = height[nx*(j) + i-1];
      h_dl = height[nx*(j-1) + i-1];
      h_u = height[nx*(j+1) + i];
      h_d = height[nx*(j-1) + i];
      h_ur = height[nx*(j+1) + i+1];
      h_r = height[nx*(j) + i+1];
      h_dr = height[nx*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H[nx*j + i] = (h_xx*(1.0+h_y*h_y) + h_yy*(1.0+h_x*h_x) - 2.0*h_xy*h_x*h_y) / (2.0 * pow((1.0+h_x*h_x+h_y*h_y), 1.5));
      K = (h_xx*h_yy - h_xy*h_xy) / (pow(1.0+h_x*h_x+h_y*h_y, 2.0));
      T[nx*j + i] = H[nx*j + i]*H[nx*j + i] - 2.0*K ;
      ctheta[nx*j + i] = -h_y/(pow(1+h_x*h_x+h_y*h_y, 0.5));
    }
  }

  for(int i=0; i<nx; i++){ //cas j=0
    H[i] = H[nx+i];
    T[i] = T[nx+i];
    ctheta[i] = ctheta[nx+i];
  }

  for(int i=0; i<nx; i++){ //cas j=ny-1
    H[nx*(ny-1) + i] = H[i];
    T[nx*(ny-1) + i] = T[i];
    ctheta[nx*(ny-1) + i] = ctheta[i];
  }

  for(int j=0; j<ny; j++){ //cas i=0
    H[nx*j] = H[nx*j+1];
    T[nx*j] = T[nx*j+1];
    ctheta[nx*j] = ctheta[nx*j+1];
  }

  for(int j=0; j<ny; j++){ //cas i=nx-1
    H[nx*j+ nx-1] = H[nx*j];
    T[nx*j + nx-1] = T[nx*j];
    ctheta[nx*j + nx-1] = ctheta[nx*j];
  }

}

void init_height_map_edge(double* H_edge_x, double* H_edge_y, double* k_x, double* k_y, double* height_x_edge, double* height_y_edge, int nx, int ny, double dx){
  double h_ul, h_u, h_ur, h_l, h, h_r, h_dl, h_d, h_dr;
  double h_x, h_xx, h_y, h_yy, h_xy;

  // x edge
  for (int j=1; j<ny-1; j++){
    for(int i=1; i<nx; i++){
      h = height_x_edge[(nx+1)*j+i];
      h_ul = height_x_edge[(nx+1)*(j+1) + i-1];
      h_l = height_x_edge[(nx+1)*(j) + i-1];
      h_dl = height_x_edge[(nx+1)*(j-1) + i-1];
      h_u = height_x_edge[(nx+1)*(j+1) + i];
      h_d = height_x_edge[(nx+1)*(j-1) + i];
      h_ur = height_x_edge[(nx+1)*(j+1) + i+1];
      h_r = height_x_edge[(nx+1)*(j) + i+1];
      h_dr = height_x_edge[(nx+1)*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H_edge_x[(nx+1)*j + i] = (h_xx*(1.0+h_y*h_y) + h_yy*(1.0+h_x*h_x) - 2.0*h_xy*h_x*h_y) / (2.0 * pow((1.0+h_x*h_x+h_y*h_y), 1.5));
      k_x[(nx+1)*j + i] = (h_xx)/(pow((1 + h_x*h_x + h_y*h_y),0.5));
    }
  }
  //y edge
  for (int j=1; j<ny; j++){
    for(int i=1; i<nx-1; i++){
      h = height_y_edge[(nx)*j+i];
      h_ul = height_y_edge[(nx)*(j+1) + i-1];
      h_l = height_y_edge[(nx)*(j) + i-1];
      h_dl = height_y_edge[(nx)*(j-1) + i-1];
      h_u = height_y_edge[(nx)*(j+1) + i];
      h_d = height_y_edge[(nx)*(j-1) + i];
      h_ur = height_y_edge[(nx)*(j+1) + i+1];
      h_r = height_y_edge[(nx)*(j) + i+1];
      h_dr = height_y_edge[(nx)*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H_edge_y[(nx)*j + i] = (h_xx*(1.0+h_y*h_y) + h_yy*(1.0+h_x*h_x) - 2.0*h_xy*h_x*h_y) / (2.0 * pow((1.0+h_x*h_x+h_y*h_y), 1.5));
      k_y[(nx)*j + i] = (h_yy)/(pow((1 + h_x*h_x + h_y*h_y),0.5));
    }
  }
  for(int i=0; i<nx; i++){ //cas j=0 pour y
    H_edge_y[i] = H_edge_y[(nx)+i];
    k_y[i] = k_y[(nx)+i];
  }
  for(int i=0; i<nx+1; i++){ //cas j=0 pour x
    H_edge_x[i] = H_edge_x[(nx+1)+i];
    k_x[i] = k_x[(nx+1)+i];
  }

  for(int i=0; i<nx; i++){ //cas j=ny pour y
    H_edge_y[nx*(ny) + i] = H_edge_y[i];
    k_y[nx*(ny) + i] = k_y[i];
  }
  for(int i=0; i<nx+1; i++){ //cas j=ny-1 pour x
    H_edge_x[(nx+1)*(ny-1) + i] = H_edge_x[i];
    k_x[(nx+1)*(ny-1) + i] = k_x[i];
  }

  for(int j=0; j<ny+1; j++){ //cas i=0 pour y
    H_edge_y[nx*j] = H_edge_y[nx*j+1];
    k_y[nx*j] = k_y[nx*j+1];
  }
  for(int j=0; j<ny; j++){ //cas i=0 pour x
    H_edge_x[(nx+1)*j] = H_edge_x[(nx+1)*j+1];
    k_x[(nx+1)*j] = k_x[(nx+1)*j+1];
  }

  for(int j=0; j<ny+1; j++){ //cas i=nx-1 pour y
    H_edge_y[nx*j+ nx-1] = H_edge_y[nx*j];
    k_y[nx*j+ nx-1] = k_y[nx*j];
  }
  for(int j=0; j<ny; j++){ //cas i=nx pour x
    H_edge_x[(nx+1)*j+ nx] = H_edge_x[(nx+1)*j];
    k_x[(nx+1)*j+ nx] = k_x[(nx+1)*j];
  }
}

void read_txt(double* height, double* height_x_edge, double* height_y_edge, char *fileName, int nx){
  int j;
  int counter;
  double test;
  int dummy;
  double deepness = 0.000002;
  FILE *file;
  file = fopen(fileName,"r");
  if (file == NULL){
      exit(1);
  }
  j=0;
  counter = 0;
  while (feof(file) == 0)
  {
    for(int i=0; i<nx; i++){
      if(counter%2 == 0){
        dummy = fscanf(file, "%lf, ", &test);
        dummy = fscanf(file, "%lf, ", &height_y_edge[nx*j+i]);
        height_y_edge[nx*j+i] = deepness*height_y_edge[nx*j+i];
      } else{
        dummy = fscanf(file, "%lf, ", &height_x_edge[(nx+1)*j+i]);
        height_x_edge[(nx+1)*j+i] = deepness * height_x_edge[(nx+1)*j+i];
        dummy = fscanf(file, "%lf, ", &height[nx*j+i]);
        height[nx*j+i] = deepness * height[nx*j+i];
      }
    }
    if(counter%2==0){
      dummy = fscanf(file, "%lf, ", &test);
    } else {
      dummy = fscanf(file, "%lf, ", &height_x_edge[(nx+1)*j+nx]);
      height_x_edge[(nx+1)*j+nx] = deepness * height_x_edge[(nx+1)*j+nx];
    }

    if(counter%2 == 1){
      j++;
    }
    counter++;
    //printf("counter = %d \n", counter);

  }
  fclose(file);
}
