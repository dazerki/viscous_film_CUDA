#ifndef _SHADER_H_
#define _SHADER_H_

#include <GL/glew.h>

const GLchar* vertexSource;
const GLchar* geometrySource;
const GLchar* fragmentSource;
const GLchar* causticSource;
const GLchar* refractionSource;
const GLchar* testSource;

GLuint vertexShader;
GLuint geometryShader;
GLuint fragmentShader;

GLuint causticShader;
GLuint refractionShader;
GLuint testShader;

GLuint shaderProgram;

GLuint causticProgram;
GLuint refractionProgram;
GLuint testProgram;

void init_shaders();
void free_shaders();
void checkGLError();

#endif
