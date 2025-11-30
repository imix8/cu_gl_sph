#pragma once
#include <cuda_runtime.h>

void initSimulation(int n_fluid);
void freeSimulation();

// New Signature: Accepts a host (CPU) pointer to fill with data
int stepSimulation(float* host_render_buffer, float dt);