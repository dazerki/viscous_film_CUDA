#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void flux_x(float *u, float *data_3D, float *data_edge, int di, int dj, int rho);

__global__ void flux_y(float *u, float *data_3D, float *data_edge, int di, int dj, int rho);

#endif
