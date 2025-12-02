/* 
    Class: ECE6122 
    Last Date Modified: 12/1/25
    Description: Class implementation file to handle the GUI for configuring simulator settings
*/

#include "SimWindow.h"
#include "SimGui.h"

// Basic constructor
SimGui::SimGui(SimWindow *window, SPHParams *params) 
{
    this->window = window;
    this->params = params;

    // --- ImGui Init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window->getWindow(), true);
    ImGui_ImplOpenGL3_Init(window->glsl_version);
}

// Get the simulator parameters
SPHParams* SimGui::getParams()
{
    return params;
}

void SimGui::createFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool SimGui::displayConfigGui(Camera *cam)
{
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
    ImGui::Begin("Detailed SPH Setup");

    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Core Settings");
    ImGui::SliderInt("Particles", &(params->particle_count), 100, 20000);
    ImGui::SliderFloat("Time Step", &(params->dt), 0.0001f, 0.01f, "%.4f");
    ImGui::SliderFloat("Visual Size", &(params->visual_radius), 0.005f, 0.05f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0, 1, 1, 1), "Physics Parameters");
    ImGui::SliderFloat("Smooth Rad (h)", &(params->h), 0.02f, 0.2f);
    ImGui::SliderFloat("Mass", &(params->mass), 0.001f, 0.1f);
    ImGui::SliderFloat("Rest Density", &(params->rest_density), 0.1f, 2000.0f);
    ImGui::SliderFloat("Stiffness (k)", &(params->stiffness), 1.0f, 50.0f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Environment");
    ImGui::SliderFloat("Gravity", &(params->gravity), -100.0f, 100.0f);
    ImGui::SliderFloat("Wall Damp", &(params->damping), -0.99f, -0.1f);
    ImGui::SliderFloat("Box Size", &(params->box_size), 0.5f, 5.0f);

    ImGui::Dummy(ImVec2(0, 20));

    bool btnHit = false;
    if (ImGui::Button("INITIALIZE & RUN", ImVec2(-1, 50)))
    {
        btnHit = true;

        // Reset Camera
        cam->cam_dist = 2.5f;
        cam->cam_yaw = -45.0f;
        cam->cam_pitch = 30.0f;
    }

    ImGui::End();
    return btnHit;
}

bool SimGui::displayRunGui(Camera *cam, int frame_count)
{
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Live Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("FPS: %.1f | N: %d", ImGui::GetIO().Framerate, frame_count);

    bool btnHit = false;
    if (ImGui::Button("STOP / CONFIG"))
    {
        btnHit = true;
    }

    if (ImGui::Button("Reset Camera View")) {
        cam->cam_dist = 2.5f;
        cam->cam_yaw = -45.0f;
        cam->cam_pitch = 30.0f;
    }

    ImGui::Separator();

    const char* btnLabel = (cam->currentColorMode == 0) ? "Color: PLASMA" : "Color: BLUE";

    if (ImGui::Button(btnLabel)) {
        cam->currentColorMode = !cam->currentColorMode;
    }

    ImGui::Text("Live Tuning (Tweak safely!)");
    ImGui::SliderFloat("Gravity", &(params->gravity), -50.0f, 50.0f);
    ImGui::SliderFloat("Stiffness", &(params->stiffness), 1.0f, 50.0f);
    ImGui::SliderFloat("Damping", &(params->damping), -1.0f, -0.1f);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1, 0, 1, 1), "Interaction (Right Click)");
    ImGui::SliderFloat("Radius", &(params->interact_radius), 0.05f, 0.5f);

    ImGui::End();
    return btnHit;
}

void SimGui::render()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}