# CUDA SPH Fluid Simulation

A real-time Smoothed Particle Hydrodynamics (SPH) fluid simulation implemented in CUDA C++. The simulation is visualized using OpenGL (instanced rendering) and features an interactive configuration GUI powered by Dear ImGui.

## Project Roadmap & Goals

1.  **Improve Boundary Conditions:** * Transition from elastic collision (coefficient of restitution) to a particle-based boundary system.
    * Implement stationary "invisible" particles to act as solid cells, allowing the SPH kernel to treat boundaries identically to fluid particles for realistic pressure handling.
2.  **Solver Optimization:** * Benchmark and optimize the SPH solver (kernel adjustments, thread/grid size tuning, algorithmic fixes).
    * *Continuous process: Use metrics from step 1 to guide optimizations.*
3.  **OpenGL Visualization (Implemented):** * Visualize particles using UV Spheres or ICO Spheres.
    * Dump CUDA calculations to a variable passed to OpenGL for rendering (replacing CSV output).
    * Input: Mesh/User defined values -> Output: Frame buffer.
4.  **Interactive GUI (Implemented):** * Integrate Dear ImGui to allow users to select input parameters (particle count, time step, radius) via a popup window before/during simulation.
5.  **CUDA-OpenGL Interop (Native Linux Goal):** * Connect OpenGL and the computational layer using CUDA Interop (`cudaGraphicsGLRegisterBuffer`).
    * Share GPU buffers directly to avoid CPU round-trips (Zero-Copy). 
    * *Note: Currently using a Host-Round-Trip method for WSL2 compatibility.*
6.  **Raytracing (Reach Goal):** * Apply a mesh to the points and implement raytracing (potentially using NVIDIA OptiX) to render realistic water surfaces instead of point sprites.

---

## Dependencies

Before setting up, ensure you have the necessary development libraries installed.

**Ubuntu / WSL2:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libglm-dev libglew-dev libglfw3-dev
```

---

## Setup & Installation

The project relies on Dear ImGui for the user interface. Follow these steps to download the dependencies and build the project.

1. Download ImGui Dependencies
```bash
# Create directory
mkdir -p imgui
cd imgui

# Download Core ImGui files
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui.h](https://raw.githubusercontent.com/ocornut/imgui/master/imgui.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/imgui.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui_draw.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/imgui_draw.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui_tables.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/imgui_tables.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui_widgets.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/imgui_widgets.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imconfig.h](https://raw.githubusercontent.com/ocornut/imgui/master/imconfig.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imgui_internal.h](https://raw.githubusercontent.com/ocornut/imgui/master/imgui_internal.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imstb_rectpack.h](https://raw.githubusercontent.com/ocornut/imgui/master/imstb_rectpack.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imstb_textedit.h](https://raw.githubusercontent.com/ocornut/imgui/master/imstb_textedit.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/imstb_truetype.h](https://raw.githubusercontent.com/ocornut/imgui/master/imstb_truetype.h)

# Download Backends (GLFW and OpenGL3)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h](https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h](https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp](https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp)
wget [https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h](https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h)

# Return to project root
cd ..
```

2. Build and Run
Once the dependencies are downloaded, compile and execute the simulation:
```bash
make clean
make
./simple_sph
```

---

## Controls
- GUI: Use the setup window to configure particle count and time step.
- Camera: Left Click + Drag to rotate. Scroll to zoom.
- Simulation: Click START to begin physics. Click STOP/RESET to re-configure.