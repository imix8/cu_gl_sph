#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "shaders.h" // <--- Include the new header
#include "sph_interop.h"

#include "SimWindow.h"
#include "SimGui.h"

// ---------------- Application State ----------------
enum AppState
{
    STATE_CONFIG,
    STATE_RUNNING
};
AppState currentState = STATE_CONFIG;

// The Master Parameter Struct
SPHParams params;

Camera cam;
float lastX = 400, lastY = 300;
bool firstMouse = true;


// ---------------- Input Callbacks ----------------

extern void ImGui_ImplGlfw_CursorPosCallback(GLFWwindow *window, double x, double y);
extern void ImGui_ImplGlfw_MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
extern void ImGui_ImplGlfw_ScrollCallback(GLFWwindow *window, double xoffset, double yoffset);

void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (currentState == STATE_RUNNING && !ImGui::GetIO().WantCaptureMouse)
    {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            float sensitivity = 0.5f;
            cam.cam_yaw += xoffset * sensitivity;
            cam.cam_pitch += yoffset * sensitivity;

            if (cam.cam_pitch > 89.0f)
                cam.cam_pitch = 89.0f;
            if (cam.cam_pitch < -89.0f)
                cam.cam_pitch = -89.0f;
        }
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    if (!ImGui::GetIO().WantCaptureMouse && currentState == STATE_RUNNING)
    {
        cam.cam_dist -= (float)yoffset * 0.1f;
        if (cam.cam_dist < 0.1f)
            cam.cam_dist = 0.1f;
        if (cam.cam_dist > 5.0f)
            cam.cam_dist = 5.0f;
    }
}

// ---------------- Mesh Utils ----------------

void createSphere(std::vector<float> &vertices,
                  std::vector<unsigned int> &indices)
{
    const int X_SEGMENTS = 12;
    const int Y_SEGMENTS = 12;
    const float PI = 3.14159265359f;
    for (int y = 0; y <= Y_SEGMENTS; ++y)
    {
        for (int x = 0; x <= X_SEGMENTS; ++x)
        {
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
    for (int y = 0; y < Y_SEGMENTS; ++y)
    {
        for (int x = 0; x < X_SEGMENTS; ++x)
        {
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x + 1);
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            indices.push_back(y * (X_SEGMENTS + 1) + x + 1);
            indices.push_back((y + 1) * (X_SEGMENTS + 1) + x + 1);
        }
    }
}

void createWireCylinder(std::vector<float>& vertices,
                        std::vector<unsigned int>& indices) {
    const int SEGMENTS = 24;
    const float PI = 3.14159265359f;

    // Create vertices for Top and Bottom Rings
    for (int i = 0; i < SEGMENTS; i++) {
        float angle = (float)i / SEGMENTS * 2.0f * PI;
        float x = cos(angle);
        float y = sin(angle);

        // Vert 2*i: Bottom (z=0)
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(0.0f);
        // Vert 2*i+1: Top (z=1)
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(1.0f);
    }

    // Create Line Indices
    for (int i = 0; i < SEGMENTS; i++) {
        int base = i * 2;
        int next = ((i + 1) % SEGMENTS) * 2;

        // Bottom Ring
        indices.push_back(base);
        indices.push_back(next);
        // Top Ring
        indices.push_back(base + 1);
        indices.push_back(next + 1);
        // Vertical Connector (every 4th segment to keep it clean)
        if (i % 4 == 0) {
            indices.push_back(base);
            indices.push_back(base + 1);
        }
    }
}

// ---------------- Main ----------------
int main()
{
    // Create a window for the program
    int window_width = 1280;
    int window_height = 800;
    SimWindow simWindow(window_width, window_height);
    GLFWwindow *window = simWindow.getWindow();

    // Init the gui backend
    SimGui gui(&simWindow, &params);
   
    // Debug info on GL driver (helps diagnose CUDA interop availability)
    const GLubyte *vendor = glGetString(GL_VENDOR);
    const GLubyte *renderer = glGetString(GL_RENDERER);
    std::cout << "GL Vendor: " << (vendor ? (const char *)vendor : "?") << "\n";
    std::cout << "GL Renderer: " << (renderer ? (const char *)renderer : "?") << "\n";

    // Bind CUDA device before any CUDA runtime calls (interop stability)
    {
        cudaError_t cErr = cudaSetDevice(0);
        if (cErr != cudaSuccess)
        {
            std::cerr << "CUDA set device failed: " << cudaGetErrorString(cErr) << std::endl;
        }
    }

    // --- Callbacks ---
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // --- Resources ---
    // NEW: Load shaders from the separate file
    GLuint program = createShaderProgram();

    // 1. SPHERE MESH
    std::vector<float> sphereVerts;
    std::vector<unsigned int> sphereIndices;
    createSphere(sphereVerts, sphereIndices);

    // 2. WIRE CYLINDER MESH (NEW)
    std::vector<float> cylVerts;
    std::vector<unsigned int> cylIndices;
    createWireCylinder(cylVerts, cylIndices);

    unsigned int VAO, VBO, EBO, VBO_Inst;
    unsigned int cylVAO, cylVBO, cylEBO;

    cudaGraphicsResource *instanceVBORes = nullptr;
    bool useInterop = true;
    
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
                          (void *)0);
    glEnableVertexAttribArray(0);

    // Instance Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    // Setup Cylinder Buffers
    glGenVertexArrays(1, &cylVAO);
    glGenBuffers(1, &cylVBO);
    glGenBuffers(1, &cylEBO);
    glBindVertexArray(cylVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cylVBO);
    glBufferData(GL_ARRAY_BUFFER, cylVerts.size() * sizeof(float),
                 cylVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cylIndices.size() * sizeof(int),
                 cylIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);
    // Enable Attr 1 for shader compatibility
    glEnableVertexAttribArray(1);

    glEnable(GL_DEPTH_TEST);

    std::vector<float> host_data; // no longer used for uploads; kept for sizing state

    // --- Loop ---
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gui.createFrame();

        // ----------------------------------------------------
        //  Configure Simulator Before Running
        // ----------------------------------------------------
        if (currentState == STATE_CONFIG && gui.displayConfigGui(&cam))
        {
            host_data.resize(params.particle_count * 4);

            glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
            glBufferData(GL_ARRAY_BUFFER,
                            params.particle_count * sizeof(float) * 4, NULL,
                            GL_DYNAMIC_DRAW);

            // Register instance buffer with CUDA for direct writes
            cudaError_t cErr = cudaGraphicsGLRegisterBuffer(&instanceVBORes, VBO_Inst, cudaGraphicsRegisterFlagsWriteDiscard);
            if (cErr != cudaSuccess)
            {
                std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(cErr) << std::endl;
                instanceVBORes = nullptr;
                useInterop = false;
            }
            else
            {
                useInterop = true;
            }

            // Initialize CUDA simulation after interop resource is set up
            initSimulation(&params);

            currentState = STATE_RUNNING;
        }

        // ----------------------------------------------------
        //  Run Simulation
        // ----------------------------------------------------
        else if (currentState == STATE_RUNNING)
        {
            // ----------------------------------------------------
            // 1. Calculate Camera Matrices
            // ----------------------------------------------------
            int width, height;
            glfwGetWindowSize(window, &width, &height);

            glm::vec3 target(params.box_size / 2.0f, params.box_size / 2.0f,
                             params.box_size / 2.0f);
            float ry = glm::radians(cam.cam_yaw);
            float rp = glm::radians(cam.cam_pitch);
            glm::vec3 pos = target + glm::vec3(cam.cam_dist * cos(rp) * cos(ry),
                                               cam.cam_dist * cos(rp) * sin(ry),
                                               cam.cam_dist * sin(rp));

            glm::mat4 view = glm::lookAt(pos, target, glm::vec3(0, 0, 1));
            glm::mat4 proj = glm::perspective(
                glm::radians(45.0f), (float)width / height, 0.1f, 100.0f);

            // ----------------------------------------------------
            // 2. Interaction Logic: Ray-Plane Intersection
            // ----------------------------------------------------
            params.is_interacting = 0;
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) ==
                    GLFW_PRESS &&
                !ImGui::GetIO().WantCaptureMouse) {
                double mx, my;
                glfwGetCursorPos(window, &mx, &my);

                // Unproject mouse at near (z=0) and far (z=1)
                glm::vec3 start =
                    glm::unProject(glm::vec3(mx, height - my, 0.0f), view, proj,
                                   glm::vec4(0, 0, width, height));
                glm::vec3 end =
                    glm::unProject(glm::vec3(mx, height - my, 1.0f), view, proj,
                                   glm::vec4(0, 0, width, height));
                glm::vec3 dir = glm::normalize(end - start);

                // Plane Equation: Z = BoxSize / 2
                // We intersect the ray with the horizontal plane in the middle
                // of the fluid
                float planeZ = params.box_size * 0.5f;
                glm::vec3 normal(0, 0, 1);
                float denom = glm::dot(dir, normal);

                if (std::abs(denom) > 0.0001f) {
                    float t =
                        glm::dot(glm::vec3(0, 0, planeZ) - start, normal) /
                        denom;
                    if (t >= 0) {
                        glm::vec3 worldPos = start + dir * t;

                        // Clamp to box bounds
                        params.interact_x =
                            glm::clamp(worldPos.x, 0.0f, params.box_size);
                        params.interact_y =
                            glm::clamp(worldPos.y, 0.0f, params.box_size);
                        // IMPORTANT: Capture the Z depth!
                        params.interact_z =
                            glm::clamp(worldPos.z, 0.0f, params.box_size);

                        params.is_interacting = 1;
                    }
                }
            }

            // ----------------------------------------------------
            // 3. Physics Step
            // ----------------------------------------------------
            float cmin = 0.0f, cmax = 1.0f;
            int count = 0;
            if (useInterop && instanceVBORes)
            {
                count = stepSimulation(instanceVBORes, &params, &cmin, &cmax);
            }
            else
            {
                // Fallback path: compute and upload via CPU
                count = stepSimulationFallback(host_data.data(), &params, &cmin, &cmax);
                glBindBuffer(GL_ARRAY_BUFFER, VBO_Inst);
                glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(float) * 4, host_data.data());
            }

            // ----------------------------------------------------
            // 4. Upload data
            // ----------------------------------------------------
            glUseProgram(program);

            // Auto-Contrast from GPU-reduced values
            glUniform1f(glGetUniformLocation(program, "vmin"), cmin);
            glUniform1f(glGetUniformLocation(program, "vmax"), cmax);
            glUniform1f(glGetUniformLocation(program, "radius"),
                        params.visual_radius);
            glUniform1i(glGetUniformLocation(program, "colorMode"),
                        cam.currentColorMode);

            glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1,
                               GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1,
                               GL_FALSE, glm::value_ptr(proj));

            glBindVertexArray(VAO);
            glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(),
                                    GL_UNSIGNED_INT, 0, count);

            // ----------------------------------------------------
            // 5. Render Interaction Cylinder (Wireframe)
            // ----------------------------------------------------
            if (params.is_interacting) {
                // Set radius uniform to the interaction radius
                glUniform1f(glGetUniformLocation(program, "radius"),
                            params.interact_radius);
                // Hack: Set vmin/vmax to 0/1, and pass a huge vMag (1000) in
                // the attribute to force the color to the max value
                // (White/Yellow)
                glUniform1f(glGetUniformLocation(program, "vmin"), 0.0f);
                glUniform1f(glGetUniformLocation(program, "vmax"), 1.0f);

                glBindVertexArray(cylVAO);

                // IMPORTANT: Disable the instanced attribute array (Loc 1)
                // so we can pass manual data via glVertexAttrib4f
                glDisableVertexAttribArray(1);
                glDisable(GL_DEPTH_TEST);  // Draw on top of fluid

                // Draw a stack of wire cylinders to visualize the column
                // The physics is a sphere, but the visual helps locate the
                // mouse in 3D
                for (int k = 0; k < 10; k++) {
                    float z = (params.box_size / 10.0f) * k;
                    // Pass: x, y, z, vMag (1000.0f for bright color)
                    glVertexAttrib4f(1, params.interact_x, params.interact_y, z,
                                     1000.0f);
                    glDrawElements(GL_LINES, cylIndices.size(), GL_UNSIGNED_INT,
                                   0);
                }

                // Restore state
                glEnable(GL_DEPTH_TEST);
                glEnableVertexAttribArray(1);
            }

            // ----------------------------------------------------
            // 6. Runtime UI
            // ----------------------------------------------------
            if (gui.displayRunGui(&cam, count))
            {
                // Stop button pressed, go back to config screen
                freeSimulation();
                if (instanceVBORes)
                {
                    cudaGraphicsUnregisterResource(instanceVBORes);
                    instanceVBORes = nullptr;
                }
                currentState = STATE_CONFIG;
            }
        }

        gui.render();
        glfwSwapBuffers(window);
    }

    if (currentState == STATE_RUNNING)
        freeSimulation();
    if (instanceVBORes)
    {
        cudaGraphicsUnregisterResource(instanceVBORes);
        instanceVBORes = nullptr;
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}