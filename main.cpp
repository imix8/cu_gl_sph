#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>  // Needed for std::min_element
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#include "sph_interop.h"

// ---------------- Camera Globals ----------------
// Python defaults: elev=30, azim=-45
float cam_dist = 2.5f;
float cam_yaw = -45.0f;
float cam_pitch = 30.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;
bool isDragging = false;  // Add drag state

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

// UPDATED: Matplotlib "Plasma" approximation
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in float vMag;

uniform float vmin;
uniform float vmax;

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    
    // Key colors of the Plasma map
    vec3 c0 = vec3(0.05, 0.03, 0.53); // Deep Purple
    vec3 c1 = vec3(0.50, 0.03, 0.62); // Purple
    vec3 c2 = vec3(0.83, 0.23, 0.48); // Pink
    vec3 c3 = vec3(0.98, 0.52, 0.19); // Orange
    vec3 c4 = vec3(0.94, 0.98, 0.13); // Yellow

    if (t < 0.25) {
        return mix(c0, c1, t * 4.0);
    } else if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) * 4.0);
    } else {
        return mix(c3, c4, (t - 0.75) * 4.0);
    }
}

void main() {
    // Dynamic Normalization: (v - min) / (max - min)
    float t = (vMag - vmin) / (vmax - vmin + 0.00001);
    vec3 col = plasma(t);
    FragColor = vec4(col, 1.0);
}
)";

// ---------------- Callbacks ----------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cam_dist -= (float)yoffset * 0.1f;
    if (cam_dist < 0.1f) cam_dist = 0.1f;
    if (cam_dist > 5.0f) cam_dist = 5.0f;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    // Only rotate if left mouse button is held
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        float sensitivity = 0.5f;
        cam_yaw += xoffset * sensitivity;
        cam_pitch += yoffset * sensitivity;

        if (cam_pitch > 89.0f) cam_pitch = 89.0f;
        if (cam_pitch < -89.0f) cam_pitch = -89.0f;
    }
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Error: " << infoLog << std::endl;
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

    GLFWwindow* window =
        glfwCreateWindow(800, 600, "SPH Simulation (Python Style)", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

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

    // Host buffer
    int N_FLUID = 1024;
    std::vector<float> host_fluid_data(N_FLUID * 4);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_Instance);
    glBufferData(GL_ARRAY_BUFFER, N_FLUID * sizeof(float) * 4, NULL,
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    initSimulation(N_FLUID);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            char title[256];
            snprintf(title, sizeof(title), "SPH Sim | FPS: %d", frameCount);
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            lastTime = currentTime;
        }

        // Matplotlib-style background (Light Gray/White)
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1. Step Physics
        int particles_to_draw = stepSimulation(host_fluid_data.data(), 0.004f);

        // 2. Upload Data to GPU
        glBindBuffer(GL_ARRAY_BUFFER, VBO_Instance);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        particles_to_draw * sizeof(float) * 4,
                        host_fluid_data.data());

        glUseProgram(program);

        // 3. Dynamic Velocity Normalization (Matches Python's auto-scale)
        float current_min = 0.0f;
        float current_max = 1.0f;

        if (particles_to_draw > 0) {
            // We need to find min/max of the 4th component (w) in the array.
            // Since std::vector is contiguous, we can iterate with stride.
            // (Simple loop for readability/safety)
            current_min = 1000.0f;
            current_max = -1000.0f;
            for (int i = 0; i < particles_to_draw; i++) {
                float v = host_fluid_data[i * 4 + 3];
                if (v < current_min) current_min = v;
                if (v > current_max) current_max = v;
            }
            // Avoid divide by zero if fluid is still
            if (current_max - current_min < 0.0001f)
                current_max = current_min + 1.0f;
        }

        glUniform1f(glGetUniformLocation(program, "vmin"), current_min);
        glUniform1f(glGetUniformLocation(program, "vmax"), current_max);

        // 4. Isometric Camera Setup (Matches Python view_init(30, -45))
        glm::vec3 target(0.5f, 0.5f, 0.5f);

        float radYaw = glm::radians(cam_yaw);
        float radPitch = glm::radians(cam_pitch);

        // Z-Up Spherical Coordinates
        float camX = target.x + cam_dist * cos(radPitch) * cos(radYaw);
        float camY = target.y + cam_dist * cos(radPitch) * sin(radYaw);
        float camZ = target.z + cam_dist * sin(radPitch);

        glm::mat4 view = glm::lookAt(glm::vec3(camX, camY, camZ), target,
                                     glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f,
                                          0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE,
                           glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1,
                           GL_FALSE, glm::value_ptr(proj));

        // Smaller radius to match Python scatter 's=20'
        glUniform1f(glGetUniformLocation(program, "radius"), 0.012f);

        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(),
                                GL_UNSIGNED_INT, 0, particles_to_draw);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    freeSimulation();
    return 0;
}