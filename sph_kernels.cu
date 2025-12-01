#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "sph_interop.h"

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        }                                                                      \
    }

const float PI = 3.14159265359f;
const int TYPE_FLUID = 0;
const int TYPE_BOUNDARY = 1;

struct Particle {
    float3 pos;
    float3 vel;
    float density;
    float pressure;
    int type;
};

// Globals
Particle* d_particles = nullptr;
float4* d_render_buffer = nullptr;
int* d_fluid_counter = nullptr;
int allocated_particles = 0;
const int BLOCK_SIZE = 256;

// Physics Helper Kernels
__device__ float poly6(float r2, float h) {
    if (r2 > h * h) return 0.0f;
    float coef = 315.0f / (64.0f * PI * powf(h, 9.0f));
    return coef * powf(h * h - r2, 3);
}

__device__ float spiky(float r, float h) {
    if (r > h) return 0.0f;
    float coef = -45.0f / (PI * powf(h, 6.0f));
    return coef * powf(h - r, 2);
}

__global__ void compute_density_pressure(Particle* particles, int num_particles,
                                         float h, float mass, float k,
                                         float rest_density) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].type == TYPE_BOUNDARY) return;
    float density = 0.0f;
    for (int j = 0; j < num_particles; ++j) {
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 < h * h) {
            float contribution = mass * poly6(r2, h);
            if (particles[j].type == TYPE_BOUNDARY) contribution *= 2.0f;
            density += contribution;
        }
    }
    particles[i].density = density;
    particles[i].pressure = k * (density - rest_density);
}

__global__ void compute_forces_and_integrate(
    Particle* particles, int num_particles, float h, float mass, float dt,
    float box_size, float damping, float gravity, float radius,
    int is_interacting, float3 interact_pos, float interact_strength,
    float interact_radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].type == TYPE_BOUNDARY) return;

    float3 f_pressure = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_gravity = make_float3(0.0f, 0.0f, gravity * particles[i].density);

    // [REMOVED] Force-based interaction logic was here.
    // We now handle it as a solid boundary constraint at the end.

    // SPH Forces (Standard)
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 < h * h && r2 > 1e-12f) {
            float r = sqrtf(r2);
            float p_i = fmaxf(0.0f, particles[i].pressure);
            float p_j, rho_j;
            if (particles[j].type == TYPE_BOUNDARY) {
                rho_j =
                    particles[i].density < 1.0f ? 1.0f : particles[i].density;
                p_j = p_i;
            } else {
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
    // SOLID INTERACTION ROD (Hard Constraint)
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
    // BOX BOUNDARIES
    // ------------------------------------------------------------------
    if (particles[i].pos.x < radius) {
        particles[i].pos.x = radius;
        if (particles[i].vel.x < 0) particles[i].vel.x *= damping;
    }
    if (particles[i].pos.x > box_size - radius) {
        particles[i].pos.x = box_size - radius;
        if (particles[i].vel.x > 0) particles[i].vel.x *= damping;
    }
    if (particles[i].pos.y < radius) {
        particles[i].pos.y = radius;
        if (particles[i].vel.y < 0) particles[i].vel.y *= damping;
    }
    if (particles[i].pos.y > box_size - radius) {
        particles[i].pos.y = box_size - radius;
        if (particles[i].vel.y > 0) particles[i].vel.y *= damping;
    }
    if (particles[i].pos.z < radius) {
        particles[i].pos.z = radius;
        if (particles[i].vel.z < 0) particles[i].vel.z *= damping;
    }
    if (particles[i].pos.z > box_size - radius) {
        particles[i].pos.z = box_size - radius;
        if (particles[i].vel.z > 0) particles[i].vel.z *= damping;
    }
}

__global__ void update_render_buffer_compact(Particle* particles,
                                             float4* vbo_pos, int n,
                                             int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (particles[i].type == TYPE_FLUID) {
        int idx = atomicAdd(counter, 1);
        float vmag = sqrtf(particles[i].vel.x * particles[i].vel.x +
                           particles[i].vel.y * particles[i].vel.y +
                           particles[i].vel.z * particles[i].vel.z);
        vbo_pos[idx] = make_float4(particles[i].pos.x, particles[i].pos.y,
                                   particles[i].pos.z, vmag);
    }
}

// Host functions unchanged
void add_boundary_face(std::vector<Particle>& list, float3 start, float3 u_dir,
                       float3 v_dir, int u_count, int v_count) {
    for (int u = 0; u < u_count; ++u) {
        for (int v = 0; v < v_count; ++v) {
            Particle p;
            p.pos = make_float3(
                start.x + u * 0.04f * u_dir.x + v * 0.04f * v_dir.x,
                start.y + u * 0.04f * u_dir.y + v * 0.04f * v_dir.y,
                start.z + u * 0.04f * u_dir.z + v * 0.04f * v_dir.z);
            p.vel = make_float3(0, 0, 0);
            p.density = 1.0f;
            p.pressure = 0;
            p.type = TYPE_BOUNDARY;
            list.push_back(p);
        }
    }
}

void initSimulation(SPHParams* params) {
    if (d_particles) cudaFree(d_particles);
    if (d_render_buffer) cudaFree(d_render_buffer);
    if (d_fluid_counter) cudaFree(d_fluid_counter);
    std::vector<Particle> host_particles;
    int n_fluid = params->particle_count;
    float init_spacing = params->h / 2.0f;
    if (init_spacing < 0.01f) init_spacing = 0.01f;
    const float init_radius = 0.25f;

    for (int i = 0; i < n_fluid; ++i) {
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
    int wall_steps = static_cast<int>(params->box_size / 0.04f) + 1;
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps,
                      wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, params->box_size),
                      make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps,
                      wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps);
    add_boundary_face(host_particles, make_float3(0, params->box_size, 0),
                      make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps);
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps);
    add_boundary_face(host_particles, make_float3(params->box_size, 0, 0),
                      make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps);

    allocated_particles = host_particles.size();
    CHECK_CUDA(
        cudaMalloc(&d_particles, allocated_particles * sizeof(Particle)));
    CHECK_CUDA(cudaMemcpy(d_particles, host_particles.data(),
                          allocated_particles * sizeof(Particle),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_render_buffer, n_fluid * sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&d_fluid_counter, sizeof(int)));
    printf("Initialized: %d Fluids, %d Total. Box: %.1f\n", n_fluid,
           allocated_particles, params->box_size);
}

int stepSimulation(float* host_render_buffer, SPHParams* params) {
    int gridSize = (allocated_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_density_pressure<<<gridSize, BLOCK_SIZE>>>(
        d_particles, allocated_particles, params->h, params->mass,
        params->stiffness, params->rest_density);
    cudaDeviceSynchronize();
    float3 mouse_pos =
        make_float3(params->interact_x, params->interact_y, params->interact_z);
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
    CHECK_CUDA(cudaMemcpy(&fluid_count, d_fluid_counter, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (fluid_count > 0) {
        CHECK_CUDA(cudaMemcpy(host_render_buffer, d_render_buffer,
                              fluid_count * sizeof(float4),
                              cudaMemcpyDeviceToHost));
    }
    return fluid_count;
}

void freeSimulation() {
    if (d_particles) cudaFree(d_particles);
    if (d_fluid_counter) cudaFree(d_fluid_counter);
    if (d_render_buffer) cudaFree(d_render_buffer);
}