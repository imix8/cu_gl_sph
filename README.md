1. Improve the boundary condition handling in the SPH solver. Right now it is an elastic collision with a coefficient of restitution.  Need to alter it so that the boundary consists of a whole bunch of "invisble" particles that remain stationary that acts as solid cells.  The reason behind this is that the boundary is no longer a wall, but particles that the SPH cuda kernel can treat in the same fashion as the moving particles.  Makes the SPH solver more realistic.
2. optimize sph sovler (benchmarking, timing and alg fixes, kernel, thread, gridsize)
3. OpenGL implementation to visualize => UV Sphere or ICO Sphere that will create the indices, vertices, faces and such that an OpenGL shader will take to actually draw the sphere (input -> mesh/user_defined values, output -> frame buffer); dunp all cuda calculations to one variable (instead of dumping to .csv) and pass it to OpenGL to visualize.
4. IM GUI => allow the user to select what input parameters they want with a window popup, before the simulation is done.
5. Connect OpenGL and computational layer with cuda interop, so that buffers created on the GPU are shared with OpenGL, passing the CPU (tell OpenGL that it is allowed to use the same space that was allocated to the GPU, one additional header file from NVIDIA).
6. (reach goal). Apply a mesh from the points, implement raytracing so that the output looks like actual water, not just points.  Look into whether or not NVIDIA's optics library is able to do this.  If this is possible, we don't even need OpenGL.

*** All throughout this process, use the metrics returned from 1. to make periodic adjustments to the sph solver to make it better***


## Setup
# 1. Create the folder
mkdir -p imgui
cd imgui

# 2. Download Core ImGui files
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui_draw.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui_tables.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui_widgets.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/imconfig.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/imgui_internal.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/imstb_rectpack.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/imstb_textedit.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/imstb_truetype.h

# 3. Download the Backends (Connects ImGui to GLFW and OpenGL)
wget https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h
wget https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp
wget https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h

# 4. Go back to your main project folder
cd ..

# 5. Build and run
make clean
make 
./simple_sph