/*
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122
    Last Date Modified: 12/2/25
    Description: CUDA Kernels for the fluid simulation using SPH
*/

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "sph_interop.h"

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        }                                                                      \
    }

const float PI = 3.14159265359f;
const int TYPE_FLUID = 0;
const int TYPE_BOUNDARY = 1;

struct Particle
{
    float3 pos;
    float3 vel;
    float density;
    float pressure;
    int type;
};

// ---------------- Globals ----------------
Particle *dParticles = nullptr;
int *dFluidCounter = nullptr;
float *dVmin = nullptr;
float *dVmax = nullptr;
float4 *dRenderBuffer = nullptr; // for CPU fallback uploads
int allocatedParticles = 0;
const int BLOCK_SIZE = 256;

// ---------------- Physics Kernels ----------------

/**
 * @brief CUDA Kernel to compute a smoothed particle
 *
 * @param  r2 is the square of the distance between the center particle and neighbor
 * @param   h is the support radius
 */
__device__ float poly6(float r2, float h)
{
    // Error Check
    if (r2 > h * h)
        return 0.0f;

    // Calculate density based on the neighbor
    float coef = 315.0f / (64.0f * PI * powf(h, 9.0f));
    return coef * powf(h * h - r2, 3);
}

/**
 * @brief CUDA Kernel to compute the magnitude of the spiky gradient
 *
 * @param r is the distance betwen the two particles
 * @param h is the influence radius
 */
__device__ float spiky(float r, float h)
{
    // Error check
    if (r > h)
        return 0.0f;

    // Calculate the magnitude of the gradient
    float coef = -45.0f / (PI * powf(h, 6.0f));
    return coef * powf(h - r, 2);
}

/**
 * @brief CUDA Kernel to compute the local density of every particle
 *
 * @param particles is an array of particle structs
 * @param numParticles is the total number of particles
 * @param h is the support radius
 * @param mass is the mass of an individual particle
 * @param k is a gas constant
 * @param restDensity is the rest density of the fluid
 */
__global__ void compute_density_pressure(Particle *particles, int numParticles,
                                         float h, float mass, float k,
                                         float restDensity)
{
    // Assign a thread to each particle
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles || particles[i].type == TYPE_BOUNDARY)
        return;

    // Give an initial density value
    float density = 0.0f;

    // Loop over all of the particles
    for (int j = 0; j < numParticles; ++j)
    {
        // Calculate the displacement
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);

        // Calculate the squared distance
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 < h * h)
        {
            float contribution = mass * poly6(r2, h);
            if (particles[j].type == TYPE_BOUNDARY)
                contribution *= 2.0f;
            density += contribution; // Add to the total density
        }
    }

    // Store the density in the array
    particles[i].density = density;
    particles[i].pressure = k * (density - restDensity);
}

/**
 * @brief CUDA Kernel to compute the forces on every particle
 *
 * @param particles is an array of particle structs
 * @param numParticles is the total number of particles
 * @param h is the support radius
 * @param mass is the mass of an individual particle
 * @param dt is the timestep
 * @param boxSize is the cubic size of the simulation
 * @param damping is the velocity damping factor for boundaries
 * @param gravity is the acceleration due to gravity
 * @param radius is the particle radius
 * @param isInteracting toggles the stirring rod on and off
 * @param interactPos is the position of the center of the rod
 * @param interactStrength is the strength of the interaction force
 * @param interactRadius is the radius of the mixing rod
 */
__global__ void compute_forces_and_integrate(
    Particle *particles, int numParticles, float h, float mass, float dt,
    float boxSize, float damping, float gravity, float radius,
    int isInteracting, float3 interactPos, float interactStrength,
    float interactRadius)
{

    // Assign each particle to its own thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles || particles[i].type == TYPE_BOUNDARY)
        return;

    // Initialize the pressure and gravity
    float3 f_pressure = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_gravity = make_float3(0.0f, 0.0f, gravity * particles[i].density);

    // Loops through each particle and calculates pressure forces
    for (int j = 0; j < numParticles; ++j)
    {
        // Skips the same particle interacting with itself
        if (i == j)
            continue;

        // Calculates displacement
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;

        // Considers neighbors only
        if (r2 < h * h && r2 > 1e-12f)
        {
            float r = sqrtf(r2);
            float p_i = fmaxf(0.0f, particles[i].pressure);
            float p_j, rho_j;

            // Handles boundary particles
            if (particles[j].type == TYPE_BOUNDARY)
            {
                rho_j =
                    particles[i].density < 1.0f ? 1.0f : particles[i].density;
                p_j = p_i;
            }
            else
            {
                p_j = particles[j].pressure;
                rho_j = particles[j].density;
            }

            // Calculate pressure
            float avg_pressure = (p_i + p_j) / (2.0f * rho_j);
            float force_mag = -mass * avg_pressure * spiky(r, h) / r;
            f_pressure.x += force_mag * rij.x;
            f_pressure.y += force_mag * rij.y;
            f_pressure.z += force_mag * rij.z;
        }
    }

    float inv_rho = 1.0f / particles[i].density;
    particles[i].vel.x += (f_pressure.x + f_gravity.x) * dt * inv_rho;
    particles[i].vel.y += (f_pressure.y + f_gravity.y) * dt * inv_rho;
    particles[i].vel.z += (f_pressure.z + f_gravity.z) * dt * inv_rho;

    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;

    // ------------------------------------------------------------------
    // Mixing rod
    // ------------------------------------------------------------------
    // We treat the rod exactly like a moving wall.
    // If a particle ends up inside, we eject it and bounce its velocity.
    if (isInteracting)
    {
        // 2D Distance Check (Infinite Cylinder)
        float2 d = make_float2(particles[i].pos.x - interactPos.x,
                               particles[i].pos.y - interactPos.y);

        float dist2 = d.x * d.x + d.y * d.y;

        // If particle is INSIDE the rod
        if (dist2 < interactRadius * interactRadius && dist2 > 1e-8f)
        {
            float dist = sqrtf(dist2);
            float2 n = make_float2(d.x / dist, d.y / dist); // Surface Normal

            // 1. Position Correction (Eject to surface)
            float overlap = interactRadius - dist;
            particles[i].pos.x += n.x * overlap;
            particles[i].pos.y += n.y * overlap;

            // 2. Velocity Reflection (Bounce)
            // Project velocity onto the normal
            float v_dot_n = particles[i].vel.x * n.x + particles[i].vel.y * n.y;

            // Only bounce if particle is moving TOWARDS the rod center
            if (v_dot_n < 0.0f)
            {
                // Calculate tangential and normal components
                float2 v_n = make_float2(v_dot_n * n.x, v_dot_n * n.y);
                float2 v_t = make_float2(particles[i].vel.x - v_n.x,
                                         particles[i].vel.y - v_n.y);

                // Apply damping to the normal component (Wall Damping)
                // 'damping' is typically negative (e.g., -0.5), so this
                // reflects and slows down
                v_n.x *= damping;
                v_n.y *= damping;

                particles[i].vel.x = v_t.x + v_n.x;
                particles[i].vel.y = v_t.y + v_n.y;
            }
        }
    }

    // ------------------------------------------------------------------
    // Boundary Box
    // ------------------------------------------------------------------

    // X-Axis
    if (particles[i].pos.x < radius)
    {
        particles[i].pos.x = radius;
        if (particles[i].vel.x < 0)
            particles[i].vel.x *= damping;
    }
    if (particles[i].pos.x > boxSize - radius)
    {
        particles[i].pos.x = boxSize - radius;
        if (particles[i].vel.x > 0)
            particles[i].vel.x *= damping;
    }

    // Y-Axis
    if (particles[i].pos.y < radius)
    {
        particles[i].pos.y = radius;
        if (particles[i].vel.y < 0)
            particles[i].vel.y *= damping;
    }
    if (particles[i].pos.y > boxSize - radius)
    {
        particles[i].pos.y = boxSize - radius;
        if (particles[i].vel.y > 0)
            particles[i].vel.y *= damping;
    }

    // Z-Axis
    if (particles[i].pos.z < radius)
    {
        particles[i].pos.z = radius;
        if (particles[i].vel.z < 0)
            particles[i].vel.z *= damping;
    }
    if (particles[i].pos.z > boxSize - radius)
    {
        particles[i].pos.z = boxSize - radius;
        if (particles[i].vel.z > 0)
            particles[i].vel.z *= damping;
    }
}

/**
 * @brief CUDA Kernel to calculate the velocity of each particles and add it back the VBO
 *
 * @param  particles list of particles in the simulation
 * @param  vboPos VBO ptr
 * @param  n  total number of particles in the sim
 * @param  counter  counts the number of particles calculated
 */
__global__ void update_render_buffer_compact(Particle *particles, float4 *vboPos, int n, int *counter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (particles[i].type == TYPE_FLUID)
    {
        int idx = atomicAdd(counter, 1);
        float vmag = sqrtf(particles[i].vel.x * particles[i].vel.x +
                           particles[i].vel.y * particles[i].vel.y +
                           particles[i].vel.z * particles[i].vel.z);
        vboPos[idx] = make_float4(particles[i].pos.x,
                                  particles[i].pos.y,
                                  particles[i].pos.z,
                                  vmag);
    }
}

// Atomic helpers for float min/max using CAS on integer representation

/**
 * @brief CUDA Kernel to calculate the min velocity of a list of points
 *
 * @param  addr  current address of point in list
 * @param  val  current val in list
 */
__device__ inline void atomicFloatMin(float *addr, float val)
{
    int *addr_i = reinterpret_cast<int *>(addr);
    int old = *addr_i, assumed;
    while (true)
    {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fminf(old_f, val);
        int new_i = __float_as_int(new_f);
        old = atomicCAS(addr_i, assumed, new_i);
        if (old == assumed)
            break;
    }
}

/**
 * @brief CUDA Kernel to calculate the max velocity of a list of points
 *
 * @param  addr  current address of point in list
 * @param  val  current val in list
 */
__device__ inline void atomicFloatMax(float *addr, float val)
{
    int *addr_i = reinterpret_cast<int *>(addr);
    int old = *addr_i, assumed;
    while (true)
    {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fmaxf(old_f, val);
        int new_i = __float_as_int(new_f);
        old = atomicCAS(addr_i, assumed, new_i);
        if (old == assumed)
            break;
    }
}

/**
 * @brief CUDA Kernel to calculate the min and max velocity values of a list of points
 *
 * @param  vboPos VBO ptr
 * @param  n  number of particles in the sim
 * @param  vmin  min velocity of a particle
 * @param  vmax  max velocity of a particle
 */
__global__ void reduce_vmin_vmax(const float4 *vboPos, int n, float *vmin, float *vmax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float v = vboPos[i].w;
    atomicFloatMin(vmin, v);
    atomicFloatMax(vmax, v);
}

// ---------------- Host Functions ----------------

/**
 * @brief  Create a face of the boundary box (just a wall of fluid points)
 *
 * @param  list list of particles to add the face particles to
 * @param  start
 * @param  uDir
 * @param  vDir
 * @param  uCount
 * @param  vCount
 */
void add_boundary_face(std::vector<Particle> &list, float3 start, float3 uDir,
                       float3 vDir, int uCount, int vCount)
{
    for (int u = 0; u < uCount; ++u)
    {
        for (int v = 0; v < vCount; ++v)
        {
            Particle p;
            p.pos = make_float3(
                start.x + u * 0.04f * uDir.x + v * 0.04f * vDir.x,
                start.y + u * 0.04f * uDir.y + v * 0.04f * vDir.y,
                start.z + u * 0.04f * uDir.z + v * 0.04f * vDir.z);
            p.vel = make_float3(0, 0, 0);
            p.density = 1.0f;
            p.pressure = 0;
            p.type = TYPE_BOUNDARY;
            list.push_back(p);
        }
    }
}

/**
 * @brief Initialize all the memory for fluid particles and boundary boxes in the simulation
 *
 * @param  params  simulation physics paramaters
 */
void initSimulation(SPHParams *params)
{
    // Free any CUDA memory if not already done so
    if (dParticles)
    {
        cudaFree(dParticles);
        dParticles = nullptr;
    }
    if (dFluidCounter)
    {
        cudaFree(dFluidCounter);
        dFluidCounter = nullptr;
    }
    if (dVmin)
    {
        cudaFree(dVmin);
        dVmin = nullptr;
    }
    if (dVmax)
    {
        cudaFree(dVmax);
        dVmax = nullptr;
    }
    if (dRenderBuffer)
    {
        cudaFree(dRenderBuffer);
        dRenderBuffer = nullptr;
    }

    std::vector<Particle> hostParticles;
    int nFluid = params->particleCount;
    float initSpacing = params->h / 2.0f;
    if (initSpacing < 0.01f)
        initSpacing = 0.01f;
    const float initRadius = 0.25f;

    // Accumulate the data for each particles in the simulation on the host
    for (int i = 0; i < nFluid; ++i)
    {
        float theta = float(i) / nFluid * 2 * PI;
        float phi = fmod(i, nFluid / 8) / (nFluid / 8) * PI;
        float rSphere = 0.9f * initRadius * cbrtf(float(i) / nFluid);
        Particle p;
        p.pos = make_float3(
            0.5f * params->boxSize + rSphere * cosf(theta) * sinf(phi),
            0.7f * params->boxSize + rSphere * cosf(phi),
            0.5f * params->boxSize + rSphere * sinf(theta) * sinf(phi));
        p.vel = make_float3(0, 0, 0);
        p.density = params->restDensity;
        p.pressure = 0;
        p.type = TYPE_FLUID;
        hostParticles.push_back(p);
    }

    // Add the boundary box walls
    int wallSteps = static_cast<int>(params->boxSize / 0.04f) + 1;
    add_boundary_face(hostParticles, make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0), wallSteps, wallSteps);
    add_boundary_face(hostParticles, make_float3(0, 0, params->boxSize), make_float3(1, 0, 0), make_float3(0, 1, 0), wallSteps, wallSteps);
    add_boundary_face(hostParticles, make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 0, 1), wallSteps, wallSteps);
    add_boundary_face(hostParticles, make_float3(0, params->boxSize, 0), make_float3(1, 0, 0), make_float3(0, 0, 1), wallSteps, wallSteps);
    add_boundary_face(hostParticles, make_float3(0, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1), wallSteps, wallSteps);
    add_boundary_face(hostParticles, make_float3(params->boxSize, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1), wallSteps, wallSteps);

    // Allocate the appropriate memory on the GPU
    allocatedParticles = hostParticles.size();
    CHECK_CUDA(cudaMalloc(&dParticles, allocatedParticles * sizeof(Particle)));
    CHECK_CUDA(cudaMemcpy(dParticles, hostParticles.data(), allocatedParticles * sizeof(Particle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&dFluidCounter, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dVmin, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dVmax, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dRenderBuffer, nFluid * sizeof(float4)));

    printf("Initialized: %d Fluids, %d Total. Box: %.1f\n", nFluid,
           allocatedParticles, params->boxSize);
}

/**
 * @brief Perform one simulation step using CUDA-OpenGL interop
 *
 * @param  instanceVBORes
 * @param  params  simulation physics paramaters
 * @param  outVmin  min velocity of all points in the buffer
 * @param  outVmax  max velocity of all points in the buffer
 *
 * @return number of fluid particles in the current sim run
 */
int stepSimulation(cudaGraphicsResource *instanceVBORes, SPHParams *params, float *outVmin, float *outVmax)
{
    // Run the pressure kernel on the points
    int gridSize = (allocatedParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_density_pressure<<<gridSize, BLOCK_SIZE>>>(
        dParticles, allocatedParticles, params->h, params->mass,
        params->stiffness, params->restDensity);
    cudaDeviceSynchronize();

    float3 mousePos = make_float3(params->interactX, params->interactY, params->interactZ);

    // Run the forces kernel on the points
    compute_forces_and_integrate<<<gridSize, BLOCK_SIZE>>>(
        dParticles, allocatedParticles, params->h, params->mass, params->dt,
        params->boxSize, params->damping, params->gravity,
        params->visualRadius, params->isInteracting, mousePos,
        params->interactStrength, params->interactRadius);
    cudaDeviceSynchronize();

    // Map OpenGL VBO and fill it directly
    if (instanceVBORes == nullptr)
    {
        // Interop not available; skip rendering
        *outVmin = 0.0f;
        *outVmax = 1.0f;
        return 0;
    }
    cudaError_t mErr = cudaGraphicsMapResources(1, &instanceVBORes);
    if (mErr != cudaSuccess)
    {
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, mErr, cudaGetErrorString(mErr));
        *outVmin = 0.0f;
        *outVmax = 1.0f;
        return 0;
    }
    size_t mapped_size = 0;
    float4 *d_vbo_ptr = nullptr;
    cudaError_t pErr = cudaGraphicsResourceGetMappedPointer((void **)&d_vbo_ptr, &mapped_size, instanceVBORes);
    if (pErr != cudaSuccess || d_vbo_ptr == nullptr)
    {
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, pErr, cudaGetErrorString(pErr));
        // Attempt to unmap if mapping succeeded
        cudaGraphicsUnmapResources(1, &instanceVBORes);
        *outVmin = 0.0f;
        *outVmax = 1.0f;
        return 0;
    }

    // Ensure mapped buffer can hold fluid particles
    size_t capacity = mapped_size / sizeof(float4);
    if (capacity < (size_t)params->particleCount)
    {
        // Avoid overflow writes
        printf("Warning: mapped VBO capacity (%zu) < particleCount (%d)\n", capacity, params->particleCount);
    }

    CHECK_CUDA(cudaMemset(dFluidCounter, 0, sizeof(int)));
    update_render_buffer_compact<<<gridSize, BLOCK_SIZE>>>(
        dParticles, d_vbo_ptr, allocatedParticles, dFluidCounter);
    cudaDeviceSynchronize();
    int fluidCount = 0;
    CHECK_CUDA(cudaMemcpy(&fluidCount, dFluidCounter, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Compute vmin/vmax on GPU, then copy back small scalars
    if (fluidCount > 0)
    {
        // Initialize vmin/vmax
        float initMin = 1e30f;
        float initMax = -1e30f;
        CHECK_CUDA(cudaMemcpy(dVmin, &initMin, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dVmax, &initMax, sizeof(float), cudaMemcpyHostToDevice));

        int reduceGrid = (fluidCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_vmin_vmax<<<reduceGrid, BLOCK_SIZE>>>(d_vbo_ptr, fluidCount, dVmin, dVmax);
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemcpy(outVmin, dVmin, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outVmax, dVmax, sizeof(float), cudaMemcpyDeviceToHost));
        if (*outVmax - *outVmin < 1e-6f)
            *outVmax = *outVmin + 1.0f;
    }
    else
    {
        *outVmin = 0.0f;
        *outVmax = 1.0f;
    }

    cudaError_t uErr = cudaGraphicsUnmapResources(1, &instanceVBORes);
    if (uErr != cudaSuccess)
    {
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, uErr, cudaGetErrorString(uErr));
    }
    return fluidCount;
}

/**
 * @brief Perform one simulation step with the CPU if CUDA-OpenGL interop was not available
 *
 * @param  hostRenderBuffer  buffer containing all the fluid points
 * @param  params  simulation physics paramaters
 * @param  outVmin  min velocity of all points in the buffer
 * @param  outVmax  max velocity of all points in the buffer
 *
 * @return number of fluid particles in the current sim run
 */
int stepSimulationFallback(float *hostRenderBuffer, SPHParams *params, float *outVmin, float *outVmax)
{
    int gridSize = (allocatedParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_density_pressure<<<gridSize, BLOCK_SIZE>>>(
        dParticles, allocatedParticles, params->h, params->mass,
        params->stiffness, params->restDensity);
    cudaDeviceSynchronize();

    float3 mousePos = make_float3(params->interactX, params->interactY, params->interactZ);

    compute_forces_and_integrate<<<gridSize, BLOCK_SIZE>>>(
        dParticles, allocatedParticles, params->h, params->mass, params->dt,
        params->boxSize, params->damping, params->gravity,
        params->visualRadius, params->isInteracting, mousePos,
        params->interactStrength, params->interactRadius);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemset(dFluidCounter, 0, sizeof(int)));
    update_render_buffer_compact<<<gridSize, BLOCK_SIZE>>>(
        dParticles, dRenderBuffer, allocatedParticles, dFluidCounter);
    cudaDeviceSynchronize();

    int fluidCount = 0;
    CHECK_CUDA(cudaMemcpy(&fluidCount, dFluidCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if (fluidCount > 0)
    {
        CHECK_CUDA(cudaMemcpy(hostRenderBuffer, dRenderBuffer, fluidCount * sizeof(float4), cudaMemcpyDeviceToHost));

        // Compute vmin/vmax on CPU from hostRenderBuffer
        float minv = 1e30f, maxv = -1e30f;
        for (int i = 0; i < fluidCount; ++i)
        {
            float v = hostRenderBuffer[i * 4 + 3];
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
        if (maxv - minv < 1e-6f)
            maxv = minv + 1.0f;
        *outVmin = minv;
        *outVmax = maxv;
    }
    else
    {
        *outVmin = 0.0f;
        *outVmax = 1.0f;
    }
    return fluidCount;
}

/**
 * @brief Free all allocated cuda memory
 */
void freeSimulation()
{
    if (dParticles)
        cudaFree(dParticles);
    if (dFluidCounter)
        cudaFree(dFluidCounter);
    if (dVmin)
        cudaFree(dVmin);
    if (dVmax)
        cudaFree(dVmax);
    if (dRenderBuffer)
        cudaFree(dRenderBuffer);
}