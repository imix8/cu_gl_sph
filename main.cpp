#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "sph_interop.h"

// ---------------- Application State ----------------
enum AppState { STATE_CONFIG, STATE_RUNNING };
AppState currentState = STATE_CONFIG;

// Parameters
int param_particles = 1024;
float param_dt = 0.004f;
float param_radius = 0.012f;

// Camera
float cam_dist = 2.5f;
float cam_yaw = -45.0f;
float cam_pitch = 30.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;

// ---------------- Shaders ----------------
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;      
layout (location = 1) in vec4 aInstance; 
uniform mat4 view;
uniform mat4 projection;
uniform float radius;
out float vMag; 
void main() {
    vec3 worldPos = aPos * radius + aInstance.xyz; 
    gl_Position = projection * view * vec4(worldPos, 1.0);
    vMag = aInstance.w;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in float vMag;
uniform float vmin;
uniform float vmax;
vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.05, 0.03, 0.53);
    vec3 c1 = vec3(0.50, 0.03, 0.62);
    vec3 c2 = vec3(0.83, 0.23, 0.48);
    vec3 c3 = vec3(0.98, 0.52, 0.19);
    vec3 c4 = vec3(0.94, 0.98, 0.13);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    else if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
    else if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    else return mix(c3, c4, (t - 0.75) * 4.0);
}
void main() {
    float t = (vMag - vmin) / (vmax - vmin + 0.00001);
    vec3 col = plasma(t);
    FragColor = vec4(col, 1.0);
}
)";

// ---------------- INPUT CALLBACKS (THE FIX) ----------------

// We must declare these externs to call ImGui's functions manually
extern void ImGui_ImplGlfw_CursorPosCallback(GLFWwindow* window, double x,
                                             double y);
extern void ImGui_ImplGlfw_MouseButtonCallback(GLFWwindow* window, int button,
                                               int action, int mods);
extern void ImGui_ImplGlfw_ScrollCallback(GLFWwindow* window, double xoffset,
                                          double yoffset);

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    // 1. Pass event to ImGui (CRITICAL FIX)
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

    // 2. Camera Logic
    // Only rotate if:
    // a) We are in RUNNING state (don't spin while configuring)
    // b) ImGui doesn't want the mouse
    // c) Left mouse button is held
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
    // 1. Pass event to ImGui (CRITICAL FIX)
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // 1. Pass event to ImGui
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    // 2. Camera Zoom
    if (!ImGui::GetIO().WantCaptureMouse && currentState == STATE_RUNNING) {
        cam_dist -= (float)yoffset * 0.1f;
        if (cam_dist < 0.1f) cam_dist = 0.1f;
        if (cam_dist > 5.0f) cam_dist = 5.0f;
    }
}

// -----------------------------------------------------------

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "SHADER ERROR: " << infoLog << std::endl;
    }
    return shader;
}

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

int main() {
    if (!glfwInit()) return -1;
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(1024, 768, "SPH Configurator", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    // --- IMGUI INIT ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // --- SET CALLBACKS (Must happen AFTER ImGui Init) ---
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);  // NEW
    glfwSetScrollCallback(window, scroll_callback);

    // Shaders & Buffers
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    std::vector<float> sphereVerts;
    std::vector<unsigned int> sphereIndices;
    createSphere(sphereVerts, sphereIndices);

    unsigned int VAO, VBO_Mesh, EBO, VBO_Instance;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO_Mesh);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &VBO_Instance);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_Mesh);
    glBufferData(GL_ARRAY_BUFFER, sphereVerts.size() * sizeof(float),
                 sphereVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sphereIndices.size() * sizeof(unsigned int),
                 sphereIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);

    std::vector<float> host_fluid_data;
    glBindBuffer(GL_ARRAY_BUFFER, VBO_Instance);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (currentState == STATE_CONFIG) {
            ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 220), ImGuiCond_FirstUseEver);
            ImGui::Begin("Simulation Setup");
            ImGui::Text("Configure Parameters");
            ImGui::Separator();
            ImGui::SliderInt("Particles", &param_particles, 100, 50000);
            ImGui::SliderFloat("Time Step", &param_dt, 0.001f, 0.01f, "%.4f");
            ImGui::SliderFloat("Sphere Size", &param_radius, 0.005f, 0.05f);
            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            if (ImGui::Button("START SIMULATION", ImVec2(-1.0f, 40.0f))) {
                initSimulation(param_particles);
                host_fluid_data.resize(param_particles * 4);
                glBindBuffer(GL_ARRAY_BUFFER, VBO_Instance);
                glBufferData(GL_ARRAY_BUFFER,
                             param_particles * sizeof(float) * 4, NULL,
                             GL_DYNAMIC_DRAW);
                currentState = STATE_RUNNING;
            }
            ImGui::End();

        } else if (currentState == STATE_RUNNING) {
            int particles_to_draw =
                stepSimulation(host_fluid_data.data(), param_dt);
            glBindBuffer(GL_ARRAY_BUFFER, VBO_Instance);
            glBufferSubData(GL_ARRAY_BUFFER, 0,
                            particles_to_draw * sizeof(float) * 4,
                            host_fluid_data.data());

            glUseProgram(program);

            float c_min = 1000.0f, c_max = -1000.0f;
            for (int i = 0; i < particles_to_draw; i++) {
                float v = host_fluid_data[i * 4 + 3];
                if (v < c_min) c_min = v;
                if (v > c_max) c_max = v;
            }
            if (c_max - c_min < 0.0001f) c_max = c_min + 1.0f;

            glUniform1f(glGetUniformLocation(program, "vmin"), c_min);
            glUniform1f(glGetUniformLocation(program, "vmax"), c_max);
            glUniform1f(glGetUniformLocation(program, "radius"), param_radius);

            glm::vec3 target(0.5f, 0.5f, 0.5f);
            float radYaw = glm::radians(cam_yaw);
            float radPitch = glm::radians(cam_pitch);
            float camX = target.x + cam_dist * cos(radPitch) * cos(radYaw);
            float camY = target.y + cam_dist * cos(radPitch) * sin(radYaw);
            float camZ = target.z + cam_dist * sin(radPitch);

            glm::mat4 view = glm::lookAt(glm::vec3(camX, camY, camZ), target,
                                         glm::vec3(0.0f, 0.0f, 1.0f));
            glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                              1024.0f / 768.0f, 0.1f, 100.0f);

            glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1,
                               GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1,
                               GL_FALSE, glm::value_ptr(proj));

            glBindVertexArray(VAO);
            glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(),
                                    GL_UNSIGNED_INT, 0, particles_to_draw);

            ImGui::SetNextWindowPos(ImVec2(10, 10));
            ImGui::Begin("Controls", NULL,
                         ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Particles: %d", particles_to_draw);
            if (ImGui::Button("STOP / RESET")) {
                freeSimulation();
                currentState = STATE_CONFIG;
            }
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