# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# compile C with /usr/bin/cc
# compile CUDA with /usr/local/cuda/bin/nvcc
C_DEFINES = -DDELTA_T=0.001 -DEPSILON=0.01f -DETA=0.0f -DZETA=5.0f

C_INCLUDES = -I/home/antoine/Documents/Viscous/GPU/src -I/home/antoine/Documents/Viscous/GPU/inc

C_FLAGS = -O3 -fopenmp

CUDA_DEFINES = -DDELTA_T=0.001 -DEPSILON=0.01f -DETA=0.0f -DZETA=5.0f

CUDA_INCLUDES = -I/home/antoine/Documents/Viscous/GPU/src -I/home/antoine/Documents/Viscous/GPU/inc

CUDA_FLAGS =  --generate-code=arch=compute_70,code=[sm_70] --generate-code=arch=compute_72,code=[compute_72]

