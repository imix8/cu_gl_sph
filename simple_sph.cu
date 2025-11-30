#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// ---------------- Simulation Constants ----------------
const int N_FLUID = 1024;
const int NUM_STEPS = 800;
const float dt = 0.002f;
const float box_size = 1.0f;
const float radius = 0.25f;

// ---------------- SPH Physics Parameters ----------------
const float h = 0.08f;
const float spacing = 0.04f;
const float mass = 0.02f;
const float rest_density = 1.0f;
const float k = 4.0f;
const float PI = 3.14159265359f;

// ---------------- Particle Types ----------------
const int TYPE_FLUID = 0;
const int TYPE_BOUNDARY = 1;

struct Particle {
    float3 pos;
    float3 vel;
    float density;
    float pressure;
    int type;
};

// ---------------- SPH Kernels ----------------

// Poly6 Kernel: Used for Density estimation
__device__ float poly6(float r2, float h) {
    if (r2 > h * h) return 0.0f;
    float coef = 315.0f / (64.0f * PI * powf(h, 9.0f));
    float term = h * h - r2;
    return coef * term * term * term;
}

// Spiky Kernel: Used for Pressure Force calculation (avoids clustering)
__device__ float spiky(float r, float h) {
    if (r > h) return 0.0f;
    float coef = -45.0f / (PI * powf(h, 6.0f));
    float term = h - r;
    return coef * term * term;
}

// ---------------- CUDA Kernels ----------------

// PASS 1: Compute Density & Pressure
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

            // Boundary handling: Double the contribution of wall particles.
            // This compensates for the lack of fluid on the other side of the
            // wall.
            if (particles[j].type == TYPE_BOUNDARY) contribution *= 2.0f;

            density += contribution;
        }
    }

    particles[i].density = density;
    particles[i].pressure = k * (density - rest_density);
}

// PASS 2: Compute Forces & Integrate
__global__ void compute_forces_and_integrate(Particle* particles,
                                             int num_particles, float h,
                                             float mass, float dt,
                                             float box_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].type == TYPE_BOUNDARY) return;

    float3 f_pressure = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_gravity = make_float3(0.0f, 0.0f, -40.0f * particles[i].density);

    // Calculate Forces
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;

        if (r2 < h * h && r2 > 1e-12f) {
            float r = sqrtf(r2);
            float p_j, rho_j;
            float p_i =
                fmaxf(0.0f, particles[i].pressure);  // Clamp negative pressure

            // Mirror Boundary Logic
            if (particles[j].type == TYPE_BOUNDARY) {
                rho_j = particles[i].density;    // Mirror density
                if (rho_j < 1.0f) rho_j = 1.0f;  // Enforce solid wall
                p_j = p_i;                       // Mirror pressure
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

    // Integrate (Euler)
    float inv_rho = 1.0f / particles[i].density;
    particles[i].vel.x += (f_pressure.x + f_gravity.x) * dt * inv_rho;
    particles[i].vel.y += (f_pressure.y + f_gravity.y) * dt * inv_rho;
    particles[i].vel.z += (f_pressure.z + f_gravity.z) * dt * inv_rho;

    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;

    // Boundary Conditions: Reflect Velocity (Bounce)
    const float DAMPING = -0.5f;  // Energy loss on bounce
    const float eps = 1e-3f;

    // X-Axis
    if (particles[i].pos.x < eps) {
        particles[i].pos.x = eps;
        particles[i].vel.x *= DAMPING;
    } else if (particles[i].pos.x > box_size - eps) {
        particles[i].pos.x = box_size - eps;
        particles[i].vel.x *= DAMPING;
    }

    // Y-Axis
    if (particles[i].pos.y < eps) {
        particles[i].pos.y = eps;
        particles[i].vel.y *= DAMPING;
    } else if (particles[i].pos.y > box_size - eps) {
        particles[i].pos.y = box_size - eps;
        particles[i].vel.y *= DAMPING;
    }

    // Z-Axis
    if (particles[i].pos.z < eps) {
        particles[i].pos.z = eps;
        particles[i].vel.z *= DAMPING;
    } else if (particles[i].pos.z > box_size - eps) {
        particles[i].pos.z = box_size - eps;
        particles[i].vel.z *= DAMPING;
    }
}

// ---------------- Host Helpers ----------------

void add_boundary_face(std::vector<Particle>& list, float3 start, float3 u_dir,
                       float3 v_dir, int u_count, int v_count, float spacing) {
    for (int u = 0; u < u_count; ++u) {
        for (int v = 0; v < v_count; ++v) {
            Particle p;
            p.pos.x = start.x + u * spacing * u_dir.x + v * spacing * v_dir.x;
            p.pos.y = start.y + u * spacing * u_dir.y + v * spacing * v_dir.y;
            p.pos.z = start.z + u * spacing * u_dir.z + v * spacing * v_dir.z;
            p.vel = make_float3(0, 0, 0);
            p.density = rest_density;
            p.pressure = 0;
            p.type = TYPE_BOUNDARY;
            list.push_back(p);
        }
    }
}

int main() {
    std::vector<Particle> host_particles;

    // 1. Initialize Fluid (Sphere drop)
    for (int i = 0; i < N_FLUID; ++i) {
        float theta = float(i) / N_FLUID * 2 * PI;
        float phi = fmod(i, N_FLUID / 8) / (N_FLUID / 8) * PI;
        float r = 0.9f * radius * cbrtf(float(i) / N_FLUID);

        Particle p;
        p.pos.x = 0.5f * box_size + r * cosf(theta) * sinf(phi);
        p.pos.y = 0.7f * box_size + r * cosf(phi);
        p.pos.z = 0.5f * box_size + r * sinf(theta) * sinf(phi);
        p.vel = make_float3(0, 0, 0);
        p.density = rest_density;
        p.pressure = 0;
        p.type = TYPE_FLUID;
        host_particles.push_back(p);
    }

    // 2. Initialize Walls
    int wall_steps = static_cast<int>(box_size / spacing) + 1;

    // Z planes
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps,
                      wall_steps, spacing);
    add_boundary_face(host_particles, make_float3(0, 0, box_size),
                      make_float3(1, 0, 0), make_float3(0, 1, 0), wall_steps,
                      wall_steps, spacing);
    // Y planes
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps, spacing);
    add_boundary_face(host_particles, make_float3(0, box_size, 0),
                      make_float3(1, 0, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps, spacing);
    // X planes
    add_boundary_face(host_particles, make_float3(0, 0, 0),
                      make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps, spacing);
    add_boundary_face(host_particles, make_float3(box_size, 0, 0),
                      make_float3(0, 1, 0), make_float3(0, 0, 1), wall_steps,
                      wall_steps, spacing);

    int total_particles = host_particles.size();
    printf("Simulation Initialized: %d Fluid, %d Total Particles\n", N_FLUID,
           total_particles);

    // 3. GPU Setup
    Particle* d_particles;
    cudaMalloc(&d_particles, total_particles * sizeof(Particle));
    cudaMemcpy(d_particles, host_particles.data(),
               total_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (total_particles + blockSize - 1) / blockSize;

    FILE* fp = fopen("positions.csv", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    // 4. Main Loop
    for (int t = 0; t < NUM_STEPS; ++t) {
        compute_density_pressure<<<gridSize, blockSize>>>(
            d_particles, total_particles, h, mass, k, rest_density);
        cudaDeviceSynchronize();

        compute_forces_and_integrate<<<gridSize, blockSize>>>(
            d_particles, total_particles, h, mass, dt, box_size);
        cudaDeviceSynchronize();

        // Copy back for I/O
        cudaMemcpy(host_particles.data(), d_particles,
                   total_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

        for (int i = 0; i < total_particles; ++i) {
            if (host_particles[i].type == TYPE_FLUID) {
                float vmag =
                    sqrtf(host_particles[i].vel.x * host_particles[i].vel.x +
                          host_particles[i].vel.y * host_particles[i].vel.y +
                          host_particles[i].vel.z * host_particles[i].vel.z);

                fprintf(fp, "%f,%f,%f,%f,%d,%d\n", host_particles[i].pos.x,
                        host_particles[i].pos.y, host_particles[i].pos.z, vmag,
                        t, i);
            }
        }
    }

    fclose(fp);
    cudaFree(d_particles);
    printf("Simulation Complete. Data saved to positions.csv\n");
    return 0;
}