#ifndef _SHADER_H_
#define _SHADER_H_

#include <GL/glew.h>

const GLchar* vertexSource;
const GLchar* geometrySource;
const GLchar* fragmentSource;
const GLchar* causticSource;
const GLchar* refractionSource;

GLuint vertexShader;
GLuint geometryShader;
GLuint fragmentShader;
GLuint shaderProgram;

void init_shaders();
void free_shaders();

#endif
