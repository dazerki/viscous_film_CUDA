// compute shaders tutorial
// Dr Anton Gerdelan <gerdela@scss.tcd.ie>
// Trinity College Dublin, Ireland
// 26 Feb 2016

#include "gl_utils.h"
#include <math.h>

GLFWwindow* window;



void stop_gl() { glfwTerminate(); }

void print_shader_info_log( GLuint shader ) {
  int max_length    = 4096;
  int actual_length = 0;
  char slog[4096];
  glGetShaderInfoLog( shader, max_length, &actual_length, slog );
  fprintf( stderr, "shader info log for GL index %u\n%s\n", shader, slog );
}

void print_program_info_log( GLuint program ) {
  int max_length    = 4096;
  int actual_length = 0;
  char plog[4096];
  glGetProgramInfoLog( program, max_length, &actual_length, plog );
  fprintf( stderr, "program info log for GL index %u\n%s\n", program, plog );
}

bool check_shader_errors( GLuint shader ) {
  GLint params = -1;
  glGetShaderiv( shader, GL_COMPILE_STATUS, &params );
  if ( GL_TRUE != params ) {
    fprintf( stderr, "ERROR: shader %u did not compile\n", shader );
    print_shader_info_log( shader );
    return false;
  }
  return true;
}

bool check_program_errors( GLuint program ) {
  GLint params = -1;
  glGetProgramiv( program, GL_LINK_STATUS, &params );
  if ( GL_TRUE != params ) {
    fprintf( stderr, "ERROR: program %u did not link\n", program );
    print_program_info_log( program );
    return false;
  }
  return true;
}

GLuint create_quad_vao() {
  GLuint vao = 0, vbo = 0;
  float verts[] = { -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, 16 * sizeof( float ), verts, GL_STATIC_DRAW );
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );
  glEnableVertexAttribArray( 0 );
  GLintptr stride = 4 * sizeof( float );
  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, NULL );
  glEnableVertexAttribArray( 1 );
  GLintptr offset = 2 * sizeof( float );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, (GLvoid*)offset );
  return vao;
}

// this is the quad's vertex shader in an ugly C string
const char* vert_shader_str =
  "#version 430\n                                                               \
layout (location = 0) in vec2 vp;\n                                           \
layout (location = 1) in vec2 vt;\n                                           \
out vec2 st;\n                                                                \
\n                                                                            \
void main () {\n                                                              \
  st = vt;\n                                                                  \
  gl_Position = vec4 (vp, 0.0, 1.0);\n                                        \
}\n";

// this is the quad's fragment shader in an ugly C string
const char* frag_shader_str =
  "#version 430\n                                                               \
in vec2 st;\n                                                                 \
uniform sampler2D img;\n                                                      \
out vec4 fc;\n                                                                \
\n                                                                            \
void main () {\n                                                              \
  fc = texture (img, st);\n                                                 \
}\n";

GLuint create_quad_program() {
  GLuint program     = glCreateProgram();
  GLuint vert_shader = glCreateShader( GL_VERTEX_SHADER );
  glShaderSource( vert_shader, 1, &vert_shader_str, NULL );
  glCompileShader( vert_shader );
  check_shader_errors( vert_shader ); // code moved to gl_utils.cpp
  glAttachShader( program, vert_shader );
  GLuint frag_shader = glCreateShader( GL_FRAGMENT_SHADER );
  glShaderSource( frag_shader, 1, &frag_shader_str, NULL );
  glCompileShader( frag_shader );
  check_shader_errors( frag_shader ); // code moved to gl_utils.cpp
  glAttachShader( program, frag_shader );
  glLinkProgram( program );
  check_program_errors( program ); // code moved to gl_utils.cpp
  return program;
}

void add_fluid(GLFWwindow* window, float* u){
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	int i = 512-floor(512*xpos/800);
	int j = floor(512*ypos/800);
	for(int k=-20; k<20; k++){
		for(int p=-20; p<20 ; p++){
			if((k*k)+(p*p)<400){
				u[512*(j+p)+(i+k)] = u[512*(j+p)+(i+k)] + 0.003f;
			}
		}
	}
}

void remove_fluid(GLFWwindow* window, float* u){
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	int i = 512-floor(512*xpos/800);
	int j = floor(512*ypos/800);
	for(int k=-11; k<11; k++){
		for(int p=-11; p<11 ; p++){
			if((k*k)+(p*p)<121){
				u[512*(j+p)+(i+k)] = 0.0f;
			}
		}
	}
}
