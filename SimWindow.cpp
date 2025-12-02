/* 
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122 
    Last Date Modified: 12/1/25
    Description: Class implimentation to house common OpenGL operations using GLFW
*/

#include "SimWindow.h"
#include <stdexcept>
#include <iostream>

/**
 * @brief  Constructor that initializes the necessary frameworks and creates a window to be stored in this class
 * 
 * @param  width  width of the simulation window
 * @param  height  height of the simulation window
 */
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
    window = glfwCreateWindow(width, height, "SPH Simulation", NULL, NULL);
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

/**
 * @brief  Get the window private member value
 * 
 * @return window of type GLFWwindow*
 */
GLFWwindow* SimWindow::getWindow()
{
    return window;
}