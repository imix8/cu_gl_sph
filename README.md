# CUDA SPH Fluid Simulation

The resources folder contains a video of the program running PACE-ICE. The resources folder also contains the report for the project.

A real-time Smoothed Particle Hydrodynamics (SPH) fluid simulation implemented in CUDA C++. The simulation is visualized using OpenGL (instanced rendering) and features an interactive configuration GUI powered by Dear ImGui.

## Project Roadmap & Goals

1. **Improve Boundary Conditions:** * Transition from elastic collision (coefficient of restitution) to a particle-based boundary system.
   * Implement stationary "invisible" particles to act as solid cells, allowing the SPH kernel to treat boundaries identically to fluid particles for realistic pressure handling.
2. **Solver Optimization:** * Benchmark and optimize the SPH solver (kernel adjustments, thread/grid size tuning, algorithmic fixes).
   * *Continuous process: Use metrics from step 1 to guide optimizations.*
3. **OpenGL Visualization (Implemented):** * Visualize particles using UV Spheres or ICO Spheres.
   * Dump CUDA calculations to a variable passed to OpenGL for rendering (replacing CSV output).
   * Input: Mesh/User defined values -> Output: Frame buffer.
4. **Interactive GUI (Implemented):** * Integrate Dear ImGui to allow users to select input parameters (particle count, time step, radius) via a popup window before/during simulation.
5. **CUDA-OpenGL Interop (Native Linux Goal):** * Connect OpenGL and the computational layer using CUDA Interop (`cudaGraphicsGLRegisterBuffer`).
   * Share GPU buffers directly to avoid CPU round-trips (Zero-Copy).
   * *Note: Currently using a Host-Round-Trip method for WSL2 compatibility.*
6. **Raytracing (Reach Goal):** * Apply a mesh to the points and implement raytracing (potentially using NVIDIA OptiX) to render realistic water surfaces instead of point sprites.

---

## Dependencies

Before setting up, ensure you have the necessary development libraries installed.

**Ubuntu / WSL2:**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libglm-dev libglew-dev libglfw3-dev
sudo apt-get install libxinerama-dev libxcursor-dev libxi-dev libxrandr-dev
```

---

## Setup & Installation

The project relies on Dear ImGui for the user interface. Follow these steps to download the dependencies and build the project.

1. Download ImGui Dependencies

```bash
make setup
```

2. Build and Run
   Once the dependencies are downloaded, compile and execute the simulation:

```bash
make clean
make
make run
```

## Building and working with CUDA-OpenGL Interop on PACE-ICE
Before provisioning yourself a desktop on PACE-ICE, make sure the "Hardware (GPU) Rendering" option is selected.
Also make sure that you are requesting a desktop with access to the GPU. Once you have access to the GUI, open the terminal
and run the following commands:
```bash
module load cuda
module load gcc/13.3.0
make -j$(nproc)
```

After the program has built, run it with `vglrun -d egl ./simple_sph`

---

## Controls

- GUI: Use the setup window to configure particle count and time step.
- Camera: Left Click + Drag to rotate. Scroll to zoom.
- Simulation: Click START to begin physics. Click STOP/RESET to re-configure.