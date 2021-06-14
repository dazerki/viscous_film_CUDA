// compute shaders tutorial
// Dr Anton Gerdelan <gerdela@scss.tcd.ie>
// Trinity College Dublin, Ireland
// 26 Feb 2016
#ifndef _UTIL_H_
#define _UTIL_H_


#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assert.h>
#include <stdio.h>


// window dimensions
#define WINDOW_W 800
#define WINDOW_H 800

// bool start_gl();
void stop_gl();
bool check_shader_errors( GLuint shader );
bool check_program_errors( GLuint program );
GLuint create_quad_vao();
GLuint create_quad_program();

// void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void add_fluid(GLFWwindow* window, float* u);

extern GLFWwindow* window;

#endif
