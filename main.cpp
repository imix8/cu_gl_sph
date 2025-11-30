#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "shaders.h"  // <--- Include the new header
#include "sph_interop.h"

// ---------------- Application State ----------------
enum AppState { STATE_CONFIG, STATE_RUNNING };
AppState currentState = STATE_CONFIG;

// The Master Parameter Struct
SPHParams params;

// Camera Globals
float cam_dist = 2.5f;
float cam_yaw = -45.0f;
float cam_pitch = 30.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;

// ---------------- Input Callbacks ----------------

extern void ImGui_ImplGlfw_CursorPosCallback(GLFWwindow* window, double x,
                                             double y);
extern void ImGui_ImplGlfw_MouseButtonCallback(GLFWwindow* window, int button,
                                               int action, int mods);
extern void ImGui_ImplGlfw_ScrollCallback(GLFWwindow* window, double xoffset,
                                          double yoffset);

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (currentState == STATE_RUNNING && !ImGui::GetIO().WantCaptureMouse) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            float sensitivity = 0.5f;
            cam_yaw += xoffset * sensitivity;
            cam_pitch += yoffset * sensitivity;

            if (cam_pitch > 89.0f) cam_pitch = 89.0f;
            if (cam_pitch < -89.0f) cam_pitch = -89.0f;
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action,
                           int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    if (!ImGui::GetIO().WantCaptureMouse && currentState == STATE_RUNNING) {
        cam_dist -= (float)yoffset * 0.1f;
        if (cam_dist < 0.1f) cam_dist = 0.1f;
        if (cam_dist > 5.0f) cam_dist = 5.0f;
    }
}

// ---------------- Mesh Utils ----------------

void createSphere(std::vector<float>& vertices,
                  std::vector<unsigned int>& indices) {
    const int X_SEGMENTS = 12;
    const int Y_SEGMENTS = 12;
    const float PI = 3.14159265359f;
    for (int y = 0; y <= Y_SEGMENTS; ++y) {
        for (int x = 0; x <= X_SEGMENTS; ++x) {
            float xSeg = (float)x / X_SEGMENTS;
            float ySeg = (float)y / Y_SEGMENTS;
            float xPos = std::cos(xSeg * 2.0f * PI) * std::sin(ySeg * PI);
            float yPos = std::cos(ySeg * PI);
            float zPos = std::sin(xSeg * 2.0f * PI) * std::sin(ySeg * PI);
            vertices.push_back(xPos);
            vertices.push_back(yPos);
            vertices.push_back(zPos);
        }
    }
    for (int y = 0; y < Y_SEGMENTS; ++y) {
        for (int x = 0; x < X_SEGMENTS; ++x) {
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x + 1);
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x + 1);
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x + 1);
        }
    }
}

// ---------------- Main ----------------
int main() {
    if (!glfwInit()) return -1;

    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(1280, 800, "SPH Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    // --- ImGui Init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // --- Callbacks ---
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // --- Resources ---
    // NEW: Load shaders from the separate file
    GLuint program = createShaderProgram();

    std::vector<float> sphereVerts;
    std::vector<unsigned int> sphereIndices;
    createSphere(sphereVerts, sphereIndices);

    unsigned int VAO, VBO, EBO, VBO_Inst;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &VBO_Inst);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVerts.size() * sizeof(float),
                 sphereVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(int),
                 sphereIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);

    // Instance Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glEnable(GL_DEPTH_TEST);

    std::vector<float> host_data;

    // --- Loop ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (currentState == STATE_CONFIG) {
            ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
            ImGui::Begin("Detailed SPH Setup");

            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Core Settings");
            ImGui::SliderInt("Particles", &params.particle_count, 100, 20000);
            ImGui::SliderFloat("Time Step", &params.dt, 0.001f, 0.01f, "%.4f");
            ImGui::SliderFloat("Visual Size", &params.visual_radius, 0.005f,
                               0.05f);

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "Physics Parameters");
            ImGui::SliderFloat("Smooth Rad (h)", &params.h, 0.02f, 0.2f);
            ImGui::SliderFloat("Mass", &params.mass, 0.001f, 0.1f);
            ImGui::SliderFloat("Rest Density", &params.rest_density, 0.1f,
                               2000.0f);
            ImGui::SliderFloat("Stiffness (k)", &params.stiffness, 1.0f, 50.0f);

            ImGui::Separator();
            ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Environment");
            ImGui::SliderFloat("Gravity", &params.gravity, -100.0f, 100.0f);
            ImGui::SliderFloat("Wall Damp", &params.damping, -0.99f, -0.1f);
            ImGui::SliderFloat("Box Size", &params.box_size, 0.5f, 5.0f);

            ImGui::Dummy(ImVec2(0, 20));

            if (ImGui::Button("INITIALIZE & RUN", ImVec2(-1, 50))) {
                initSimulation(&params);
                host_data.resize(params.particle_count * 4);

                glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
                glBufferData(GL_ARRAY_BUFFER,
                             params.particle_count * sizeof(float) * 4, NULL,
                             GL_DYNAMIC_DRAW);

                // Reset Camera
                cam_dist = 2.5f;
                cam_yaw = -45.0f;
                cam_pitch = 30.0f;

                currentState = STATE_RUNNING;
            }
            ImGui::End();

        } else if (currentState == STATE_RUNNING) {
            // 1. Physics Step
            int count = stepSimulation(host_data.data(), &params);

            // 2. Upload Data
            glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
            glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(float) * 4,
                            host_data.data());

            // 3. Render
            glUseProgram(program);

            // Auto-Contrast
            float cmin = 1000.0f, cmax = -1000.0f;
            for (int i = 0; i < count; i++) {
                float v = host_data[i * 4 + 3];
                if (v < cmin) cmin = v;
                if (v > cmax) cmax = v;
            }
            if (cmax - cmin < 0.001f) cmax = cmin + 1.0f;

            glUniform1f(glGetUniformLocation(program, "vmin"), cmin);
            glUniform1f(glGetUniformLocation(program, "vmax"), cmax);
            glUniform1f(glGetUniformLocation(program, "radius"),
                        params.visual_radius);

            // Camera Matrices
            glm::vec3 target(params.box_size / 2.0f, params.box_size / 2.0f,
                             params.box_size / 2.0f);
            float ry = glm::radians(cam_yaw);
            float rp = glm::radians(cam_pitch);
            glm::vec3 pos = target + glm::vec3(cam_dist * cos(rp) * cos(ry),
                                               cam_dist * cos(rp) * sin(ry),
                                               cam_dist * sin(rp));

            glm::mat4 view = glm::lookAt(pos, target, glm::vec3(0, 0, 1));
            glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                              1280.0f / 800.0f, 0.1f, 100.0f);

            glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1,
                               GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1,
                               GL_FALSE, glm::value_ptr(proj));

            glBindVertexArray(VAO);
            glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(),
                                    GL_UNSIGNED_INT, 0, count);

            // 4. Runtime UI
            ImGui::SetNextWindowPos(ImVec2(10, 10));
            ImGui::Begin("Live Controls", NULL,
                         ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("FPS: %.1f | N: %d", ImGui::GetIO().Framerate, count);

            if (ImGui::Button("STOP / CONFIG")) {
                freeSimulation();
                currentState = STATE_CONFIG;
            }

            ImGui::Separator();
            if (ImGui::Button("Reset Camera View")) {
                cam_dist = 2.5f;
                cam_yaw = -45.0f;
                cam_pitch = 30.0f;
            }

            ImGui::Text("Live Tuning (Tweak safely!)");
            ImGui::SliderFloat("Gravity", &params.gravity, -50.0f, 50.0f);
            ImGui::SliderFloat("Stiffness", &params.stiffness, 1.0f, 50.0f);
            ImGui::SliderFloat("Damping", &params.damping, -1.0f, -0.1f);
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (currentState == STATE_RUNNING) freeSimulation();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}