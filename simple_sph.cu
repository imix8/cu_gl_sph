// simple_sph.cu
#include <cmath>
#include <cstdio>

// Simulation parameters
const int N = 1024;         // Number of particles
const int steps = 800;      // Number of timesteps
const float dt = 0.002;     // Time step
const float box = 1.0;      // Size of the simulation cube
const float radius = 0.25;  // Initial droplet radius

// SPH kernel parameters
const float h = 0.08;  // Smoothing length
const float mass = 0.02;
const float rest_density = 1.0;
const float k = 4.0;  // Pressure constant

// const int M_PI = 3.14159265358979323846;

struct Particle {
    float3 pos, vel;
    float density, pressure;
};

__device__ float poly6(float r2, float h) {
    if (r2 > h * h) return 0;
    float coef = 315.0f / (64.0f * M_PI * pow(h, 9));
    return coef * pow(h * h - r2, 3);
}

__device__ float spiky(float r, float h) {
    if (r > h) return 0;
    float coef = -45.0f / (M_PI * pow(h, 6));
    return coef * pow(h - r, 2);
}

// <<<1, N>>> launch for simplicity
__global__ void sph_step(Particle* particles, float mass, float h, float k,
                         float rest_density, float dt, float box) {
    int i = threadIdx.x;
    // Compute density and pressure
    float density = 0;
    for (int j = 0; j < N; ++j) {
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        density += mass * poly6(r2, h);
    }
    particles[i].density = density;
    particles[i].pressure = k * (density - rest_density);
    __syncthreads();

    // Compute forces
    float3 fpressure = make_float3(0, 0, 0);
    float3 fgravity = make_float3(0, 0, -40.0f * particles[i].density);
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float3 rij = make_float3(particles[i].pos.x - particles[j].pos.x,
                                 particles[i].pos.y - particles[j].pos.y,
                                 particles[i].pos.z - particles[j].pos.z);
        float r = sqrtf(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
        if (r < h && r > 1e-6) {
            float avg_pressure =
                (particles[i].pressure + particles[j].pressure) /
                (2.0f * particles[j].density);
            float coeff = -mass * avg_pressure * spiky(r, h) / r;
            fpressure.x += coeff * rij.x;
            fpressure.y += coeff * rij.y;
            fpressure.z += coeff * rij.z;
        }
    }
    // Update velocity and position
    particles[i].vel.x +=
        (fpressure.x + fgravity.x) * dt / particles[i].density;
    particles[i].vel.y +=
        (fpressure.y + fgravity.y) * dt / particles[i].density;
    particles[i].vel.z +=
        (fpressure.z + fgravity.z) * dt / particles[i].density;

    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;

    float restitution = 0.5f;  // 0=stick, 1=perfect bounce

    // X boundaries
    if (particles[i].pos.x <= 0.0f && particles[i].vel.x < 0.0f)
        particles[i].vel.x *= -restitution;
    if (particles[i].pos.x >= box && particles[i].vel.x > 0.0f)
        particles[i].vel.x *= -restitution;
    // Y boundaries
    if (particles[i].pos.y <= 0.0f && particles[i].vel.y < 0.0f)
        particles[i].vel.y *= -restitution;
    if (particles[i].pos.y >= box && particles[i].vel.y > 0.0f)
        particles[i].vel.y *= -restitution;
    // Z boundaries
    if (particles[i].pos.z <= 0.0f && particles[i].vel.z < 0.0f)
        particles[i].vel.z *= -restitution;
    if (particles[i].pos.z >= box && particles[i].vel.z > 0.0f)
        particles[i].vel.z *= -restitution;

    // Clamp after bounce to keep within bounds
    particles[i].pos.x = fmaxf(0.0f, fminf(box, particles[i].pos.x));
    particles[i].pos.y = fmaxf(0.0f, fminf(box, particles[i].pos.y));
    particles[i].pos.z = fmaxf(0.0f, fminf(box, particles[i].pos.z));
}

int main() {
    Particle* h_particles = new Particle[N];
    // Spawn droplet in upper third, centered
    for (int i = 0; i < N; ++i) {
        float theta = float(i) / N * 2 * M_PI;
        float phi = fmod(i, N / 8) / (N / 8) * M_PI;
        float r = 0.9f * radius * cbrtf(float(i) / N);  // Spread in droplet
        h_particles[i].pos.x = 0.5f * box + r * cosf(theta) * sinf(phi);
        h_particles[i].pos.y = 0.7f * box + r * cosf(phi);
        h_particles[i].pos.z = 0.5f * box + r * sinf(theta) * sinf(phi);
        h_particles[i].vel = make_float3(0, 0, 0);
    }
    Particle* d_particles;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle),
               cudaMemcpyHostToDevice);

    FILE* fp = fopen("positions.csv", "w");
    for (int t = 0; t < steps; ++t) {
        sph_step<<<1, N>>>(d_particles, mass, h, k, rest_density, dt, box);
        cudaMemcpy(h_particles, d_particles, N * sizeof(Particle),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i)
            fprintf(fp, "%f,%f,%f,%f,%d,%d\n", h_particles[i].pos.x,
                    h_particles[i].pos.y, h_particles[i].pos.z,
                    sqrt(h_particles[i].vel.x * h_particles[i].vel.x +
                         h_particles[i].vel.y * h_particles[i].vel.y +
                         h_particles[i].vel.z * h_particles[i].vel.z),
                    t, i);
    }
    fclose(fp);
    cudaFree(d_particles);
    delete[] h_particles;
    return 0;
}
