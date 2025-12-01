#pragma once
#include <cuda_runtime.h>

// A container for all our tunable parameters
struct SPHParams {
    int particle_count = 1024;
    float dt = 0.004f;
    float visual_radius = 0.012f;

    // Physics
    float h = 0.08f;     // Smoothing radius
    float mass = 0.02f;  // Particle mass
    float rest_density = 1.0f;
    float stiffness = 4.0f;  // Gas constant (k)

    // Environment
    float damping = -0.5f;  // Wall bounce energy loss
    float gravity = -40.0f;
    float box_size = 1.0f;

    // --- NEW: Interaction Parameters ---
    int is_interacting = 0;  // 0 = false, 1 = true
    float interact_x = 0.0f;
    float interact_y = 0.0f;
    float interact_z = 0.0f;
    float interact_strength = 500.0f;  // Force magnitude
    float interact_radius = 0.25f;     // Radius of influence
};

// Initialize memory
void initSimulation(SPHParams* params);

// Cleanup
void freeSimulation();

// Run Step
int stepSimulation(float* host_render_buffer, SPHParams* params);