/* 
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122 
    Last Date Modified: 12/1/25
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
Particle *d_particles = nullptr;
int *d_fluid_counter = nullptr;
float *d_vmin = nullptr;
float *d_vmax = nullptr;
float4 *d_render_buffer = nullptr; // for CPU fallback uploads
int allocated_particles = 0;
const int BLOCK_SIZE = 256;

// ---------------- Physics Kernels ----------------

/**
 * @brief CUDA Kernel to 
 * 
 * @param  
 * @param   
 */
__device__ float poly6(float r2, float h)
{
    if (r2 > h * h)
        return 0.0f;
    float coef = 315.0f / (64.0f * PI * powf(h, 9.0f));
    return coef * powf(h * h - r2, 3);
}

/**
 * @brief CUDA Kernel to
 * 
 * @param  
 * @param   
 */
__device__ float spiky(float r, float h)
{
    if (r > h)
        return 0.0f;
    float coef = -45.0f / (PI * powf(h, 6.0f));
    return coef * powf(h - r, 2);
}

/**
 * @brief CUDA Kernel to compute the local density of every particle
 * 
 * @param  
 * @param   
 */
__global__ void compute_density_pressure(Particle *particles, int num_particles,
                                         float h, float mass, float k,
                                         float rest_density)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].type == TYPE_BOUNDARY) return;
    float density = 0.0f;
    for (int j = 0; j < num_particles; ++j)
    {
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 < h * h)
        {
            float contribution = mass * poly6(r2, h);
            if (particles[j].type == TYPE_BOUNDARY)
                contribution *= 2.0f;
            density += contribution;
        }
    }
    particles[i].density = density;
    particles[i].pressure = k * (density - rest_density);
}

/**
 * @brief CUDA Kernel to compute the forces on every particle
 * 
 * @param  
 * @param  
 */
__global__ void compute_forces_and_integrate(
    Particle* particles, int num_particles, float h, float mass, float dt,
    float box_size, float damping, float gravity, float radius,
    int is_interacting, float3 interact_pos, float interact_strength,
    float interact_radius)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].type == TYPE_BOUNDARY)
        return;

    float3 f_pressure = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_gravity = make_float3(0.0f, 0.0f, gravity * particles[i].density);

    for (int j = 0; j < num_particles; ++j)
    {
        if (i == j)
            continue;
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;

        if (r2 < h * h && r2 > 1e-12f)
        {
            float r = sqrtf(r2);
            float p_i = fmaxf(0.0f, particles[i].pressure);
            float p_j, rho_j;

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
    if (is_interacting) {
        // 2D Distance Check (Infinite Cylinder)
        float2 d = make_float2(particles[i].pos.x - interact_pos.x,
                               particles[i].pos.y - interact_pos.y);

        float dist2 = d.x * d.x + d.y * d.y;

        // If particle is INSIDE the rod
        if (dist2 < interact_radius * interact_radius && dist2 > 1e-8f) {
            float dist = sqrtf(dist2);
            float2 n = make_float2(d.x / dist, d.y / dist);  // Surface Normal

            // 1. Position Correction (Eject to surface)
            float overlap = interact_radius - dist;
            particles[i].pos.x += n.x * overlap;
            particles[i].pos.y += n.y * overlap;

            // 2. Velocity Reflection (Bounce)
            // Project velocity onto the normal
            float v_dot_n = particles[i].vel.x * n.x + particles[i].vel.y * n.y;

            // Only bounce if particle is moving TOWARDS the rod center
            if (v_dot_n < 0.0f) {
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
    if (particles[i].pos.x < radius) {
        particles[i].pos.x = radius;
        if (particles[i].vel.x < 0)
            particles[i].vel.x *= damping;
    }
    if (particles[i].pos.x > box_size - radius)
    {
        particles[i].pos.x = box_size - radius;
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
    if (particles[i].pos.y > box_size - radius)
    {
        particles[i].pos.y = box_size - radius;
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
    if (particles[i].pos.z > box_size - radius)
    {
        particles[i].pos.z = box_size - radius;
        if (particles[i].vel.z > 0)
            particles[i].vel.z *= damping;
    }
}

/**
 * @brief CUDA Kernel to calculate the velocity of each particles and add it back the VBO
 * 
 * @param  particles list of particles in the simulation
 * @param  vbo_pos VBO ptr
 * @param  n  total number of particles in the sim
 * @param  counter  counts the number of particles calculated
 */
__global__ void update_render_buffer_compact(Particle *particles, float4 *vbo_pos, int n, int *counter)
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
        vbo_pos[idx] = make_float4(particles[i].pos.x,
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
 * @param  vbo_pos VBO ptr
 * @param  n  number of particles in the sim
 * @param  vmin  min velocity of a particle
 * @param  vmax  max velocity of a particle
 */
__global__ void reduce_vmin_vmax(const float4 *vbo_pos, int n, float *vmin, float *vmax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float v = vbo_pos[i].w;
    atomicFloatMin(vmin, v);
    atomicFloatMax(vmax, v);
}

// ---------------- Host Functions ----------------

/**
 * @brief  Create a face of the boundary box (just a wall of fluid points)
 * 
 * @param  list list of particles to add the face particles to
 * @param  start 
 * @param  u_dir
 * @param  v_dir
 * @param  u_count
 * @param  v_count
 */
void add_boundary_face(std::vector<Particle> &list, float3 start, float3 u_dir,
                       float3 v_dir, int u_count, int v_count)
{
    for (int u = 0; u < u_count; ++u)
    {
        for (int v = 0; v < v_count; ++v)
        {
            Particle p;
            p.pos = make_float3(
                start.x + u * 0.04f * u_dir.x + v * 0.04f * v_dir.x,
                start.y + u * 0.04f * u_dir.y + v * 0.04f * v_dir.y,
                start.z + u * 0.04f * u_dir.z + v * 0.04f * v_dir.z
            );
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
    if (d_particles)
    {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    if (d_fluid_counter)
    {
        cudaFree(d_fluid_counter);
        d_fluid_counter = nullptr;
    }
    if (d_vmin)
    {
        cudaFree(d_vmin);
        d_vmin = nullptr;
    }
    if (d_vmax)
    {
        cudaFree(d_vmax);
        d_vmax = nullptr;
    }
    if (d_render_buffer)
    {
        cudaFree(d_render_buffer);
        d_render_buffer = nullptr;
    }

    std::vector<Particle> host_particles;
    int n_fluid = params->particle_count;
    float init_spacing = params->h / 2.0f;
    if (init_spacing < 0.01f) init_spacing = 0.01f;
    const float init_radius = 0.25f;

    // Accumulate the data for each particles in the simulation on the host
    for (int i = 0; i < n_fluid; ++i)
    {
        float theta = float(i) / n_fluid * 2 * PI;
        float phi = fmod(i, n_fluid / 8) / (n_fluid / 8) * PI;
        float r_sphere = 0.9f * init_radius * cbrtf(float(i) / n_fluid);
        Particle p;
        p.pos = make_float3(
            0.5f * params->box_size + r_sphere * cosf(theta) * sinf(phi),
            0.7f * params->box_size + r_sphere * cosf(phi),
            0.5f * params->box_size + r_sphere * sinf(theta) * sinf(phi));
        p.vel = make_float3(0, 0, 0);
        p.density = params->rest_density;
        p.pressure = 0;
        p.type = TYPE_FLUID;
        host_particles.push_back(p);
    }

    // Add the boundary box walls
    int wall_steps = static_cast<int>(params->box_size / 0.04f) + 1;
    add_boundary_face(host_particles, make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps, wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, params->box_size), make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps, wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, 0),make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps, wall_steps);
    add_boundary_face(host_particles, make_float3(0, params->box_size, 0), make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps, wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps, wall_steps);
    add_boundary_face(host_particles, make_float3(params->box_size, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps, wall_steps);

    // Allocate the appropriate memory on the GPU
    allocated_particles = host_particles.size();
    CHECK_CUDA(cudaMalloc(&d_particles, allocated_particles * sizeof(Particle)));
    CHECK_CUDA(cudaMemcpy(d_particles, host_particles.data(), allocated_particles * sizeof(Particle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_fluid_counter, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_vmin, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vmax, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_render_buffer, n_fluid * sizeof(float4)));

    printf("Initialized: %d Fluids, %d Total. Box: %.1f\n", n_fluid,
           allocated_particles, params->box_size);
}

/**
 * @brief Perform one simulation step using CUDA-OpenGL interop
 * 
 * @param  instanceVBORes   
 * @param  params  simulation physics paramaters
 * @param  out_vmin  min velocity of all points in the buffer
 * @param  out_vmax  max velocity of all points in the buffer
 * 
 * @return number of fluid particles in the current sim run
 */
int stepSimulation(cudaGraphicsResource *instanceVBORes, SPHParams *params, float *out_vmin, float *out_vmax)
{
    // Run the pressure kernel on the points
    int gridSize = (allocated_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_density_pressure<<<gridSize, BLOCK_SIZE>>>(
        d_particles, allocated_particles, params->h, params->mass,
        params->stiffness, params->rest_density
    );
    cudaDeviceSynchronize();

    float3 mouse_pos = make_float3(params->interact_x, params->interact_y, params->interact_z);
    
    // Run the forces kernel on the points
    compute_forces_and_integrate<<<gridSize, BLOCK_SIZE>>>(
        d_particles, allocated_particles, params->h, params->mass, params->dt,
        params->box_size, params->damping, params->gravity,
        params->visual_radius, params->is_interacting, mouse_pos,
        params->interact_strength, params->interact_radius
    );
    cudaDeviceSynchronize();

    // Map OpenGL VBO and fill it directly
    if (instanceVBORes == nullptr)
    {
        // Interop not available; skip rendering
        *out_vmin = 0.0f;
        *out_vmax = 1.0f;
        return 0;
    }
    cudaError_t mErr = cudaGraphicsMapResources(1, &instanceVBORes);
    if (mErr != cudaSuccess)
    {
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, mErr, cudaGetErrorString(mErr));
        *out_vmin = 0.0f;
        *out_vmax = 1.0f;
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
        *out_vmin = 0.0f;
        *out_vmax = 1.0f;
        return 0;
    }

    // Ensure mapped buffer can hold fluid particles
    size_t capacity = mapped_size / sizeof(float4);
    if (capacity < (size_t)params->particle_count)
    {
        // Avoid overflow writes
        printf("Warning: mapped VBO capacity (%zu) < particle_count (%d)\n", capacity, params->particle_count);
    }

    CHECK_CUDA(cudaMemset(d_fluid_counter, 0, sizeof(int)));
    update_render_buffer_compact<<<gridSize, BLOCK_SIZE>>>(
        d_particles, d_vbo_ptr, allocated_particles, d_fluid_counter);
    cudaDeviceSynchronize();
    int fluid_count = 0;
    CHECK_CUDA(cudaMemcpy(&fluid_count, d_fluid_counter, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Compute vmin/vmax on GPU, then copy back small scalars
    if (fluid_count > 0)
    {
        // Initialize vmin/vmax
        float init_min = 1e30f;
        float init_max = -1e30f;
        CHECK_CUDA(cudaMemcpy(d_vmin, &init_min, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vmax, &init_max, sizeof(float), cudaMemcpyHostToDevice));

        int reduceGrid = (fluid_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_vmin_vmax<<<reduceGrid, BLOCK_SIZE>>>(d_vbo_ptr, fluid_count, d_vmin, d_vmax);
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemcpy(out_vmin, d_vmin, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(out_vmax, d_vmax, sizeof(float), cudaMemcpyDeviceToHost));
        if (*out_vmax - *out_vmin < 1e-6f)
            *out_vmax = *out_vmin + 1.0f;
    }
    else
    {
        *out_vmin = 0.0f;
        *out_vmax = 1.0f;
    }

    cudaError_t uErr = cudaGraphicsUnmapResources(1, &instanceVBORes);
    if (uErr != cudaSuccess)
    {
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, uErr, cudaGetErrorString(uErr));
    }
    return fluid_count;
}

/**
 * @brief Perform one simulation step with the CPU if CUDA-OpenGL interop was not available
 * 
 * @param  host_render_buffer  buffer containing all the fluid points
 * @param  params  simulation physics paramaters
 * @param  out_vmin  min velocity of all points in the buffer
 * @param  out_vmax  max velocity of all points in the buffer
 * 
 * @return number of fluid particles in the current sim run
 */
int stepSimulationFallback(float *host_render_buffer, SPHParams *params, float *out_vmin, float *out_vmax)
{
    int gridSize = (allocated_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_density_pressure<<<gridSize, BLOCK_SIZE>>>(
        d_particles, allocated_particles, params->h, params->mass,
        params->stiffness, params->rest_density);
    cudaDeviceSynchronize();

    float3 mouse_pos = make_float3(params->interact_x, params->interact_y, params->interact_z);

    compute_forces_and_integrate<<<gridSize, BLOCK_SIZE>>>(
        d_particles, allocated_particles, params->h, params->mass, params->dt,
        params->box_size, params->damping, params->gravity,
        params->visual_radius, params->is_interacting, mouse_pos,
        params->interact_strength, params->interact_radius);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemset(d_fluid_counter, 0, sizeof(int)));
    update_render_buffer_compact<<<gridSize, BLOCK_SIZE>>>(
        d_particles, d_render_buffer, allocated_particles, d_fluid_counter);
    cudaDeviceSynchronize();

    int fluid_count = 0;
    CHECK_CUDA(cudaMemcpy(&fluid_count, d_fluid_counter, sizeof(int), cudaMemcpyDeviceToHost));
    if (fluid_count > 0)
    {
        CHECK_CUDA(cudaMemcpy(host_render_buffer, d_render_buffer, fluid_count * sizeof(float4), cudaMemcpyDeviceToHost));

        // Compute vmin/vmax on CPU from host_render_buffer
        float minv = 1e30f, maxv = -1e30f;
        for (int i = 0; i < fluid_count; ++i)
        {
            float v = host_render_buffer[i * 4 + 3];
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
        if (maxv - minv < 1e-6f)
            maxv = minv + 1.0f;
        *out_vmin = minv;
        *out_vmax = maxv;
    }
    else
    {
        *out_vmin = 0.0f;
        *out_vmax = 1.0f;
    }
    return fluid_count;
}

/**
 * @brief Free all allocated cuda memory
 */
void freeSimulation()
{
    if (d_particles)
        cudaFree(d_particles);
    if (d_fluid_counter)
        cudaFree(d_fluid_counter);
    if (d_vmin)
        cudaFree(d_vmin);
    if (d_vmax)
        cudaFree(d_vmax);
    if (d_render_buffer)
        cudaFree(d_render_buffer);
}