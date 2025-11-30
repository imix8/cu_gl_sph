#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

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