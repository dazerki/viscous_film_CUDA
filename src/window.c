#include "window.h"


/*
 *  GLFW, GLEW initialisation
 */
GLFWwindow *init_window() {
    // Init GLFW & window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "Viscous film", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Callbacks
    glfwSetKeyCallback(window, key_callback);

    // Init GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    return window;
}



/*
 *  Callback for key presses
 */
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        printf("Spacebar pressed !\n");
    }
}
