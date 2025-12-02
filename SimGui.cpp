/*
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122
    Last Date Modified: 12/2/25
    Description: Class implementation file to handle the GUI for configuring simulator settings
*/

#include "SimWindow.h"
#include "SimGui.h"

/**
 * @brief  Constructor that initializes the gui and stores the necessary pointers
 *
 * @param  window  SimWindow object for the program
 * @param  params  Simulator parameter struct
 * @param  cam     Simulator camera settings struct
 */
SimGui::SimGui(SimWindow *window, SPHParams *params, Camera *cam)
{
    this->window = window;
    this->params = params;
    this->cam = cam;

    // --- ImGui Init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window->getWindow(), true);
    ImGui_ImplOpenGL3_Init(window->glsl_version);
}

/**
 * @brief  Getter for the simulator parameters struct
 *
 * @return ptr to the sim params struct
 */
SPHParams *SimGui::getParams()
{
    return params;
}

/**
 * @brief  Initialize a new ImGui, OpenGL, and GLFW frame for the program
 */
void SimGui::createFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

/**
 * @brief  Display the pre-run config gui for the user
 *
 * When the user changes a slider in the gui, the params struct for the simulator is updated
 *
 * @return true if the user pressed the RUN button, else false
 */
bool SimGui::displayConfigGui()
{
    // Create all the UI elements and connect them to values (ie. sliders for config values)
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
    ImGui::Begin("Detailed SPH Setup");

    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Core Settings");
    ImGui::SliderInt("Particles", &(params->particleCount), 100, 20000);
    ImGui::SliderFloat("Time Step", &(params->dt), 0.0001f, 0.01f, "%.4f");
    ImGui::SliderFloat("Visual Size", &(params->visualRadius), 0.005f, 0.05f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0, 1, 1, 1), "Physics Parameters");
    ImGui::SliderFloat("Smooth Rad (h)", &(params->h), 0.02f, 0.2f);
    ImGui::SliderFloat("Mass", &(params->mass), 0.001f, 0.1f);
    ImGui::SliderFloat("Rest Density", &(params->restDensity), 0.1f, 2000.0f);
    ImGui::SliderFloat("Stiffness (k)", &(params->stiffness), 1.0f, 50.0f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Environment");
    ImGui::SliderFloat("Gravity", &(params->gravity), -100.0f, 100.0f);
    ImGui::SliderFloat("Wall Damp", &(params->damping), -0.99f, -0.1f);
    ImGui::SliderFloat("Box Size", &(params->boxSize), 0.5f, 5.0f);

    ImGui::Dummy(ImVec2(0, 20));

    // Check if the user presses the run button
    bool btnHit = false;
    if (ImGui::Button("INITIALIZE & RUN", ImVec2(-1, 50)))
    {
        btnHit = true;

        // Reset Camera
        cam->camDist = 2.5f;
        cam->camYaw = -45.0f;
        cam->camPitch = 30.0f;
    }

    ImGui::End();
    return btnHit;
}

/**
 * @brief  Display the runtime config gui for the user
 *
 * When the user changes a slider in the gui, the params struct for the simulator is updated
 * @param frameCount is the number of frames
 *
 * @return true if the user pressed the STOP button, else false
 */
bool SimGui::displayRunGui(int frameCount)
{
    // Create the UI elements and display some live data about the simulation
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Live Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("FPS: %.1f | N: %d", ImGui::GetIO().Framerate, frameCount);

    // Check if the user presses the STOP button
    bool btnHit = false;
    if (ImGui::Button("STOP / CONFIG"))
    {
        btnHit = true;
    }

    // Button to reset the camera view
    if (ImGui::Button("Reset Camera View"))
    {
        cam->camDist = 2.5f;
        cam->camYaw = -45.0f;
        cam->camPitch = 30.0f;
    }

    ImGui::Separator();

    // Button to change the color of the points
    const char *colorBtn = (cam->currentColorMode == 0) ? "Color: PLASMA" : "Color: BLUE";
    if (ImGui::Button(colorBtn))
    {
        cam->currentColorMode = !cam->currentColorMode;
    }

    // More UI config elements
    ImGui::Text("Live Tuning (Tweak safely!)");
    ImGui::SliderFloat("Gravity", &(params->gravity), -50.0f, 50.0f);
    ImGui::SliderFloat("Stiffness", &(params->stiffness), 1.0f, 50.0f);
    ImGui::SliderFloat("Damping", &(params->damping), -1.0f, -0.1f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1, 0, 1, 1), "Interaction (Right Click)");
    ImGui::SliderFloat("Radius", &(params->interactRadius), 0.05f, 0.5f);

    ImGui::End();
    return btnHit;
}

/**
 * @brief  Render the current gui in the window
 */
void SimGui::render()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}