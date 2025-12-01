/* 
    Class: ECE6122 
    Last Date Modified: 12/1/25
    Description: Class to house common OpenGL operations using GLFW and input callback functions
*/

#include "SimWindow.h"
#include <stdexcept>
#include <iostream>

// Constructor that initializes the necessary frameworks and creates a window to be stored in this class 
SimWindow::SimWindow(int width, int height) 
{
    // Init glfw
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        throw std::runtime_error("");
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a new window for the sim to run in
    window = glfwCreateWindow(widht, height, "SPH Simulation", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        throw std::runtime_error("");
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewErr) << std::endl;
        glfwTerminate();
        throw std::runtime_error("");
    }
}

// Get the window
GLFWwindow* SimWindow::getWindow()
{
    return window;
}