# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

# Compilers
NVCC := nvcc
CXX  := g++

# Output Executable Name
TARGET := simple_sph

# CUDA Install Path (Adjust if your CUDA is installed elsewhere)
CUDA_PATH ?= /usr/local/cuda

# --------------------------------------------------------
# Flags & Libraries
# --------------------------------------------------------

# CUDA Flags
# -c: Compile to object file
# -O3: Optimize
# -std=c++14: C++ Standard
NVCCFLAGS := -O3 -std=c++14

# C++ Flags (Host)
# We need to tell C++ where to find CUDA headers (cuda_runtime.h)
CXXFLAGS := -O3 -std=c++14 -Wall -I$(CUDA_PATH)/include

# Linker Flags
# Link against OpenGL, GLEW, GLFW, and CUDA Runtime
LDFLAGS := -lGL -lGLEW -lglfw -lcudart

# --------------------------------------------------------
# Files
# --------------------------------------------------------

# Source files
CUDA_SRC := sph_kernels.cu
CPP_SRC  := main.cpp
HEADERS  := sph_interop.h

# Object files (intermediate build files)
CUDA_OBJ := sph_kernels.o
CPP_OBJ  := main.o

# --------------------------------------------------------
# Build Rules
# --------------------------------------------------------

all: $(TARGET)

# Link Step: Combine C++ and CUDA objects into the final executable
# We use NVCC to link because it handles CUDA library paths automatically
$(TARGET): $(CPP_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C++ (Host Code)
$(CPP_OBJ): $(CPP_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA (Device Code)
$(CUDA_OBJ): $(CUDA_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Cleanup
clean:
	rm -f $(TARGET) $(CPP_OBJ) $(CUDA_OBJ) positions.csv

.PHONY: all clean