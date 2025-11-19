1. Improve the boundary condition handling in the SPH solver.
2. optimize sph sovler (benchmarking, timing and alg fixes, kernel, thread, gridsize)
3. OpenGL implementation to visualize => UV Sphere or ICO Sphere that will create the indices, vertices, faces and such that an OpenGL shader will take to actually draw the sphere (input -> mesh/user_defined values, output -> frame buffer); dunp all cuda calculations to one variable (instead of dumping to .csv) and pass it to OpenGL to visualize.
4. IM GUI => allow the user to select what input parameters they want with a window popup, before the simulation is done.
5. Connect OpenGL and computational layer with cuda interop, so that buffers created on the GPU are shared with OpenGL, passing the CPU (tell OpenGL that it is allowed to use the same space that was allocated to the GPU, one additional header file from NVIDIA).
6. (reach goal). Apply a mesh from the points, implement raytracing so that the output looks like actual water, not just points.  Look into whether or not NVIDIA's optics library is able to do this.  If this is possible, we don't even need OpenGL.

*** All throughout this process, use the metrics returned from 1. to make periodic adjustments to the sph solver to make it better***