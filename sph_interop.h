#pragma once
#include <cuda_runtime.h>

// Initialize simulation with N particles
void initSimulation(int n_fluid);

// Cleanup memory
void freeSimulation();

// Run one physics step and copy data to the host buffer
// Returns the number of active particles
int stepSimulation(float* host_render_buffer, float dt);