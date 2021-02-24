#ifndef _VISCOUS_H_
#define _VISCOUS_H_

void initialization(float* u, int nx, int ny, float h, int choice);

void double_circle(float* u, int nx, int ny, float value);

void circle(float* u, int nx, int ny, float value);

void gaussians(float* u, int nx, int ny, float h);

void simple_gaussian(float* u, int nx, int ny, float h);

void big_line(float* u, int nx, int ny, float value);

void merging_gaussian(float* u, int nx, int ny, float h);

void init_surface(double* H, double* T, double* ctheta, int nx, int ny, double h);

void init_surface_height_map(double* H, double* T, double* ctheta, double* height, int nx, int ny, double dx);

double x_deriv(double h_r, double h_l, double dx);

double y_deriv(double h_u, double h_d, double dy);

double xx_deriv(double h_r, double h_l, double h, double dx);

double yy_deriv(double h_u, double h_d, double h, double dy);

double xy_deriv(double h_ur, double h_dr, double h_ul, double h_dl, double dx, double dy);

void read_txt(double* height,double* height_x_edge, double* height_y_edge, char *fileName, int nx);

void init_height_map_edge(double* H_edge_x, double* H_edge_y, double* k_x, double* k_y, double* height_x_edge, double* height_y_edge, int nx, int ny, double dx);

#endif
