/*
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122
    Last Date Modified: 12/2/25
    Description: Header file for simulation utilities
*/

#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Camera Globals
struct Camera
{
    float camDist = 2.5f;
    float camYaw = -45.0f;
    float camPitch = 30.0f;
    int currentColorMode = 0; // 0 = Plasma, 1 = Blue
};

// A container for all our tunable parameters
struct SPHParams
{
    int particleCount = 1024;
    float dt = 0.004f;
    float visualRadius = 0.012f;

    // Physics
    float h = 0.08f;    // Smoothing radius
    float mass = 0.02f; // Particle mass
    float restDensity = 1.0f;
    float stiffness = 4.0f; // Gas constant (k)

    // Environment
    float damping = -0.5f; // Wall bounce energy loss
    float gravity = -40.0f;
    float boxSize = 1.0f;

    // Interaction Parameters
    int isInteracting = 0;
    float interactX = 0.0f;
    float interactY = 0.0f;
    float interactZ = 0.0f;
    float interactStrength = 500.0f; // Force magnitude
    float interactRadius = 0.25f;    // Radius of influence
};

// Initialize memory
void initSimulation(SPHParams *params);

// Cleanup
void freeSimulation();

// Run Step using CUDA-OpenGL interop. Writes directly into mapped instance VBO.
// Returns number of fluid particles to render, and outputs vmin/vmax of velocity magnitude.
int stepSimulation(cudaGraphicsResource *instanceVBORes, SPHParams *params, float *outVmin, float *outVmax);

// Fallback: Run step and copy to host buffer, for CPU->OpenGL upload when interop is unavailable
int stepSimulationFallback(float *host_render_buffer, SPHParams *params, float *outVmin, float *outVmax);