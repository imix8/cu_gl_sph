/* 
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

public:
    // Basic constructor
    SimGui(SimWindow *window, SPHParams *params);

    // Get the simulator parameters
    SPHParams* getParams();

    void createFrame();

    bool displayConfigGui(Camera *cam);

    bool displayRunGui(Camera *cam, int frame_count);

    void render();
};