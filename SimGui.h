/* 
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122
    Last Date Modified: 12/1/25
    Description: Class header file to handle the GUI for configuring simulator settings
*/
#pragma once

#include "sph_interop.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class SimGui
{
private:
    // Private member data
    SimWindow *window;
    SPHParams *params;
    Camera    *cam;

public:
    // Basic constructor
    SimGui(SimWindow *window, SPHParams *params, Camera *cam);

    // Getter for the simulator parameters struct
    SPHParams* getParams();

    // Initialize a new ImGui, OpenGL, and GLFW frame for the program
    void createFrame();

    // Display the pre-run config gui for the user
    bool displayConfigGui();

    // Display the runtime config gui for the user
    bool displayRunGui(int frame_count);

    // Render the gui
    void render();
};