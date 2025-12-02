/* 
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122 
    Last Date Modified: 12/1/25
    Description: Header file for simulation utilities
*/

#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Camera Globals
struct Camera {
    float cam_dist = 2.5f;
    float cam_yaw = -45.0f;
    float cam_pitch = 30.0f;
    int currentColorMode = 0;  // 0 = Plasma, 1 = Blue
};

// A container for all our tunable parameters
struct SPHParams
{
    int particle_count = 1024;
    float dt = 0.004f;
    float visual_radius = 0.012f;

    // Physics
    float h = 0.08f;    // Smoothing radius
    float mass = 0.02f; // Particle mass
    float rest_density = 1.0f;
    float stiffness = 4.0f; // Gas constant (k)

    // Environment
    float damping = -0.5f; // Wall bounce energy loss
    float gravity = -40.0f;
    float box_size = 1.0f;

    // Interaction Parameters
    int is_interacting = 0;
    float interact_x = 0.0f;
    float interact_y = 0.0f;
    float interact_z = 0.0f;
    float interact_strength = 500.0f;  // Force magnitude
    float interact_radius = 0.25f;     // Radius of influence
};

// Initialize memory
void initSimulation(SPHParams *params);

// Cleanup
void freeSimulation();

// Run Step using CUDA-OpenGL interop. Writes directly into mapped instance VBO.
// Returns number of fluid particles to render, and outputs vmin/vmax of velocity magnitude.
int stepSimulation(cudaGraphicsResource *instanceVBORes, SPHParams *params, float *out_vmin, float *out_vmax);

// Fallback: Run step and copy to host buffer, for CPU->OpenGL upload when interop is unavailable
int stepSimulationFallback(float *host_render_buffer, SPHParams *params, float *out_vmin, float *out_vmax);