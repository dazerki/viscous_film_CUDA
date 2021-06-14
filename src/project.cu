
extern "C" {
  #include "viscous.h"
}

#include "kernel.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <cuda.h>

// compute shaders tutorial
// Dr Anton Gerdelan <gerdela@scss.tcd.ie>
// Trinity College Dublin, Ireland
// 26 Feb 2016. latest v 2 Mar 2016

#include "gl_utils.h"

// this is the compute shader in an ugly C string
const char* compute_shader_str_caustic =
  "#version 430\n                                                             \
layout (local_size_x = 1, local_size_y = 1) in;\n                             \
layout (rgba32f, binding = 0) uniform image2D data_in;\n                      \
layout (rgba32f, binding = 2) uniform image2D caustic1_in;\n                  \
layout (rgba32f, binding = 3) uniform image2D caustic2_in;\n                  \
layout (rgba32f, binding = 4) uniform image2D caustic3_in;\n                  \
layout (rgba32f, binding = 5) uniform image2D caustic4_in;\n                  \
\n                                                                            \
#define N 13 \n                                                               \
#define N_HALF 6 \n                                                           \
\n                                                                            \
const vec3 L = vec3(0.09901475,  0.09901475, -0.99014754); \n                 \
\n                                                                            \
const float hRest = .1; \n                                                    \
\n                                                                            \
const float nAir = 1.000277; \n                                               \
const float nWater = 1.330; \n                                                \
\n                                                                            \
vec3 getRefractedLightDirection(vec3 n, vec3 L) \n                            \
{ \n                                                                          \
  \n                                                                          \
  float cosTheta1 = dot(n, normalize(L)); \n                                  \
\n                                                                            \
  float refRatio = nAir / nWater; \n                                          \
\n                                                                            \
  float sinTheta2 = refRatio * sqrt(1. - cosTheta1 * cosTheta1); \n           \
  float cosTheta2 = sqrt(1. - sinTheta2 * sinTheta2); \n                      \
\n                                                                            \
  return refRatio * L + (cosTheta1 * refRatio - cosTheta2) * n; \n            \
} \n                                                                          \
\n                                                                            \
vec2 getGroundIntersection(vec2 waterIncidentPoint){ \n                       \
  ivec2 waterIncidentPoint_ij = ivec2( floor(waterIncidentPoint*vec2(512., 512.))); \n   \
\n                                                                            \
  vec3 n = imageLoad(data_in, waterIncidentPoint_ij).rgb; \n                     \
\n                                                                            \
  vec3 Ltag = getRefractedLightDirection(vec3(0., 0., 1.), L); \n                            \
\n                                                                            \
  float alpha = imageLoad(data_in, waterIncidentPoint_ij).a / Ltag.z; \n         \
\n                                                                            \
  return waterIncidentPoint + alpha * Ltag.xy; \n                             \
} \n                                                                          \
\n                                                                            \
void main() \n                                                                \
{ \n                                                                          \
    ivec2 ij = ivec2(gl_GlobalInvocationID.xy); \n                            \
    vec2 p = ij/vec2(512., 512.); \n                                          \
    vec2 h = vec2(1.,1.) / vec2(512., 512.); \n                               \
  \n                                                                          \
    float intensity[N]; \n                                                    \
    for ( int i=0; i<N; i++ ) intensity[i] = 0.; \n                           \
\n                                                                            \
    vec2 P_G = p; \n                                                          \
    vec3 n = imageLoad(data_in, ij).rgb; \n                   \
    vec3 Ltag = getRefractedLightDirection(vec3(0., 0., 1.), L); \n                          \
    float alpha = hRest / Ltag.z; \n                                          \
    vec2 P_C = P_G - alpha * Ltag.xy; \n                                      \
\n                                                                            \
    float P_Gy[N]; \n                                                         \
    for ( int i=-N_HALF; i<=N_HALF; i++ ) P_Gy[i + N_HALF] = P_G.y + float(i) * h.y; \n  \
    \n                                                                        \
    for ( int i=0; i<N; i++ ) { \n                                            \
        \n                                                                    \
        vec2 pN = P_C + float(i - N_HALF) * vec2(h.x, 0); \n                  \
        vec2 intersection = getGroundIntersection(pN); \n                     \
        \n                                                                    \
        float ax = max(0., h.x - abs(P_G.x - intersection.x)) / h.x; \n       \
        \n                                                                    \
        for ( int j=0; j<N; j++ ) { \n                                        \
            \n                                                                \
            float ay = max(0., h.y - abs(P_Gy[j] - intersection.y)) / h.y; \n \
            \n                                                                \
            intensity[j] += ax*ay; \n                                         \
        } \n                                                                  \
    } \n                                                                      \
    \n                                                                        \
    imageStore(caustic1_in, ij, vec4( intensity[0], intensity[1], intensity[2], intensity[3] ) );   \n    \
    imageStore(caustic2_in, ij, vec4( intensity[4], intensity[5], intensity[6], intensity[7] ) );   \n    \
    imageStore(caustic3_in, ij, vec4( intensity[8], intensity[9], intensity[10], intensity[11] ) ); \n    \
    imageStore(caustic4_in, ij, vec4( intensity[12], 0., 0., 0.) ); \n                                    \
} \n ";


const char* compute_shader_str_refrac =
  "#version 430\n                                                             \
layout (local_size_x = 1, local_size_y = 1) in;\n                             \
layout (rgba32f, binding = 0) uniform image2D data_in;\n                      \
layout (rgba32f, binding = 1) uniform image2D img_output;\n                   \
layout (rgba32f, binding = 2) uniform image2D caustic1_in;\n                  \
layout (rgba32f, binding = 3) uniform image2D caustic2_in;\n                  \
layout (rgba32f, binding = 4) uniform image2D caustic3_in;\n                  \
layout (rgba32f, binding = 5) uniform image2D caustic4_in;\n                  \
layout (rgba32f, binding = 6) uniform image2D ground_in;\n                    \
\n                                                                            \
const vec3 L = vec3(0.09901475,  0.09901475, -0.99014754);\n                  \
\n                                                                            \
const float nAir = 1.000277; \n                                               \
\n                                                                            \
const float fluidRefractiveIndex = 1.54;\n                                          \
const vec3 fluidColor = vec3(0.3, 0.15, 0.); \n                                                     \
const vec2 fluidClarity = vec2(0.1, 0.5); \n                                                   \
\n                                                                            \
vec2 getGroundIntersection(vec2 fluidIncidentPoint){ \n                       \
  vec2 p = fluidIncidentPoint; \n                                             \
  ivec2 fluidIncidentPoint_ij = ivec2( floor(fluidIncidentPoint*vec2(512., 512.))); \n   \
  vec2 h = vec2(1.,1.) / vec2(512., 512.); \n                                 \
  ivec2 ij = ivec2(gl_GlobalInvocationID.xy); \n                               \
\n                                                                            \
  vec3 n = imageLoad(data_in, ij).rgb; \n                     \
\n                                                                            \
  float cosTheta1 = dot(n, L); \n                                             \
\n                                                                            \
  float refRatio = nAir / fluidRefractiveIndex; \n                            \
  \n                                                                          \
  float sinTheta2 = refRatio * sqrt(1. - cosTheta1 * cosTheta1); \n           \
  float cosTheta2 = sqrt(1. - sinTheta2 * sinTheta2); \n                      \
  \n                                                                          \
  vec3 Ltag = refRatio * L + (cosTheta1 * refRatio - cosTheta2) * n; \n       \
\n                                                                            \
  float alpha = imageLoad(data_in, ij).a / Ltag.z;\n                                           \
\n                                                                            \
  return p + alpha * Ltag.xy; \n                                              \
}\n                                                                           \
\n                                                                            \
void main(){ \n                                                               \
  ivec2 ij = ivec2(gl_GlobalInvocationID.xy); \n                                \
  vec2 pos = ij / vec2(512., 512.); \n                                        \
\n                                                                            \
  vec2 h = vec2(1.,1.) / vec2(512., 512.); \n                                 \
  vec2 groundPoint = getGroundIntersection(pos); \n                           \
  ivec2 p = ivec2( floor(groundPoint*vec2(512., 512.))); \n      \
  float illumination = 0.; \n                                                 \
  illumination += imageLoad(caustic1_in, p + ivec2(0, -6)).r; \n               \
  illumination += imageLoad(caustic1_in, p + ivec2(0, -5)).g; \n               \
  illumination += imageLoad(caustic1_in, p + ivec2(0, -4)).b; \n               \
  illumination += imageLoad(caustic1_in, p + ivec2(0, -3)).a; \n               \
  illumination += imageLoad(caustic2_in, p + ivec2(0, -2)).r; \n               \
  illumination += imageLoad(caustic2_in, p + ivec2(0, -1)).g; \n               \
  illumination += imageLoad(caustic2_in, p).b; \n                             \
  illumination += imageLoad(caustic2_in, p + ivec2(0, 1)).a;  \n               \
  illumination += imageLoad(caustic3_in, p + ivec2(0, 2)).r;  \n               \
  illumination += imageLoad(caustic3_in, p + ivec2(0, 3)).g;  \n               \
  illumination += imageLoad(caustic3_in, p + ivec2(0, 4)).b;  \n               \
  illumination += imageLoad(caustic3_in, p + ivec2(0, 5)).a;  \n               \
  illumination += imageLoad(caustic4_in, p + ivec2(0, 6)).r;  \n               \
  illumination = max(illumination - .8, 0.); \n                               \
  vec3 groundColor = vec3(1., 1., 1.); \n                          \
  float height = imageLoad(data_in, ij).a; \n                                 \
  float depth = max(0., min((height - fluidClarity.x) / (fluidClarity.y - fluidClarity.x), 1.)); \n                         \
  imageStore(img_output, ij, vec4(((1. - depth) * groundColor + depth * fluidColor) + illumination * fluidColor, 1.)); \n   \
  if (imageLoad(data_in, ij).a == 0.) { \n                                    \
      imageStore(img_output, ij, vec4(0., 0., 0., 1.)); \n                    \
  } \n                                                                        \
}\n";


int main() {
  ( start_gl() ); // just starts a 4.3 GL context+window

  // set up shaders and geometry for full-screen quad
  // moved code to gl_utils.cpp
  GLuint quad_vao     = create_quad_vao();
  GLuint quad_program = create_quad_program();

  GLuint caustic_program = 0;
  { // create the compute shader
    GLuint caustic_shader = glCreateShader( GL_COMPUTE_SHADER );
    glShaderSource( caustic_shader, 1, &compute_shader_str_caustic, NULL );
    glCompileShader( caustic_shader );
    ( check_shader_errors( caustic_shader ) ); // code moved to gl_utils.cpp
    caustic_program = glCreateProgram();
    glAttachShader( caustic_program, caustic_shader );
    glLinkProgram( caustic_program );
    ( check_program_errors( caustic_program ) ); // code moved to gl_utils.cpp
  }

  GLuint refrac_program = 0;
  { // create the compute shader
    GLuint refrac_shader = glCreateShader( GL_COMPUTE_SHADER );
    glShaderSource( refrac_shader, 1, &compute_shader_str_refrac, NULL );
    glCompileShader( refrac_shader );
    ( check_shader_errors( refrac_shader ) ); // code moved to gl_utils.cpp
    refrac_program = glCreateProgram();
    glAttachShader( refrac_program, refrac_shader );
    glLinkProgram( refrac_program );
    ( check_program_errors( refrac_program ) ); // code moved to gl_utils.cpp
  }

  // texture handle and dimensions
  GLuint tex_data = 0;
  int tex_w = 512, tex_h = 512;
  { // create the texture
    glGenTextures( 1, &tex_data );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, tex_data );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 0, tex_data, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_outputColor = 0;
  { // create the texture
    glGenTextures( 1, &tex_outputColor );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, tex_outputColor );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 1, tex_outputColor, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_caustic1 = 0;
  { // create the texture
    glGenTextures( 1, &tex_caustic1 );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_2D, tex_caustic1 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 2, tex_caustic1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_caustic2 = 0;
  { // create the texture
    glGenTextures( 1, &tex_caustic2 );
    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_2D, tex_caustic2);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 3, tex_caustic2, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_caustic3 = 0;
  { // create the texture
    glGenTextures( 1, &tex_caustic3 );
    glActiveTexture( GL_TEXTURE4 );
    glBindTexture( GL_TEXTURE_2D, tex_caustic3);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 4, tex_caustic3, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_caustic4 = 0;
  { // create the texture
    glGenTextures( 1, &tex_caustic4 );
    glActiveTexture( GL_TEXTURE5 );
    glBindTexture( GL_TEXTURE_2D, tex_caustic4);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 5, tex_caustic4, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }

  // texture handle and dimensions
  GLuint tex_ground = 0;
  { // create the texture
    glGenTextures( 1, &tex_ground );
    glActiveTexture( GL_TEXTURE6 );
    glBindTexture( GL_TEXTURE_2D, tex_ground);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 6, tex_ground, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
  }


  { // query up the workgroups
    int work_grp_size[3], work_grp_inv;
    // maximum global work group (total work in a dispatch)
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_size[0] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_size[1] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_size[2] );
    printf( "max global (total) work group size x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2] );
    // maximum local work group (one shader's slice)
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2] );
    printf( "max local (in one shader) work group sizes x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2] );
    // maximum compute shader invocations (x * y * z)
    glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv );
    printf( "max computer shader invocations %i\n", work_grp_inv );
  }

  int nx = 512;
	int ny = 512;
	float h = 1.0f/nx ;
	int size = nx*ny;

  float* u = (float*)calloc(size, sizeof(float));
  float* normals = (float*)calloc(3*size, sizeof(float));
  float* data_rgba = (float*)calloc(4*size, sizeof(float));

	float *u_gpu;
  float *normals_gpu;

  size_t memSize = size*sizeof(float);

	cudaMalloc( (void**)&u_gpu, memSize );
  cudaMalloc( (void**)&normals_gpu, 3*memSize );

	//init
	initialization(u, nx, ny, h, 3);


	cudaMemcpy( u_gpu, u, memSize, cudaMemcpyHostToDevice );
  cudaMemcpy( normals_gpu, normals, 3*memSize, cudaMemcpyHostToDevice );

  int Nblocks = (nx*nx + 255)/256;
  int Nthreads = 256;

	int n_passe = 10;

  while ( !glfwWindowShouldClose( window ) ) { // drawing loop
    for(int p=0; p<n_passe; p++){
  		for(int rho=0; rho<4; rho++){
  			flux_x<<<Nblocks, Nthreads>>>(u_gpu, rho);
  		}

  		for(int rho=0; rho<4; rho++){
  			flux_y<<<Nblocks, Nthreads>>>(u_gpu, rho);
  		}

      // glfwPollEvents();
  		// if(drag){
      //   cudaMemcpy( u, u_gpu, size*sizeof(float), cudaMemcpyDeviceToHost );
  		// 	add_fluid(window);
      //   cudaMemcpy( u_gpu, u, memSize, cudaMemcpyHostToDevice );
  		// }
  	}
    normal<<<Nblocks, Nthreads>>>(u_gpu, normals_gpu, nx);

  	cudaMemcpy( u, u_gpu, size*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( normals, normals_gpu, 3*size*sizeof(float), cudaMemcpyDeviceToHost );

    for(int i=0; i<nx*ny; i++){
      data_rgba[4*i]=normals[3*nx*ny-3*i];
      data_rgba[4*i+1]=normals[3*nx*ny-(3*i+1)];
      data_rgba[4*i+2]=normals[3*nx*ny-(3*i+2)];
      data_rgba[4*i+3]=u[nx*ny-i];
    }


    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, tex_data );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, &data_rgba[0] );
    glBindImageTexture( 0, tex_data, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );

    {                                          // launch compute shaders!
      glUseProgram( caustic_program );
      glDispatchCompute( (GLuint)tex_w, (GLuint)tex_h, 1 );
    }
    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

    {                                          // launch compute shaders!
      glUseProgram( refrac_program );
      glDispatchCompute( (GLuint)tex_w, (GLuint)tex_h, 1 );
    }


    // prevent sampling befor all writes to image are done
    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

    glClear( GL_COLOR_BUFFER_BIT );
    glUseProgram( quad_program );
    glBindVertexArray( quad_vao );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, tex_outputColor );
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

    glfwPollEvents();
    if ( GLFW_PRESS == glfwGetKey( window, GLFW_KEY_ESCAPE ) ) { glfwSetWindowShouldClose( window, 1 ); }
    glfwSwapBuffers( window );
  }

  stop_gl(); // stop glfw, close window

  free(u);
	cudaFree(u_gpu);

	printf("\n *Happy computer sound* \n");
  return 0;
}
