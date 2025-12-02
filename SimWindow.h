/* 
    Class: ECE6122 
    Last Date Modified: 12/1/25
    Description: Class to house common OpenGL operations using GLFW and input callback functions
*/
#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "sph_interop.h"

class SimWindow
{
private:
    // Private member data
    GLFWwindow *window;
    SPHParams *params;

public:
    // Public member data
    const char *glsl_version = "#version 330";

    // Create a glfwWindow for the program and initialize it
    SimWindow(int width, int height);

    // Get the window
    GLFWwindow* getWindow();
};