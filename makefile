# Compilers
NVCC := nvcc
CXX  := g++

# Output
TARGET := simple_sph

# Flags
CUDA_PATH ?= /usr/local/cuda
NVCCFLAGS := -O3 -std=c++14
# Added -I./imgui
CXXFLAGS  := -O3 -std=c++14 -Wall -I$(CUDA_PATH)/include -I./imgui -I/usr/local/include

# Libraries
LDFLAGS := -lGL -lGLEW -lglfw -lcudart

# Files
CUDA_SRC := sph_kernels.cu
CPP_SRC  := main.cpp
# ImGui Sources
IMGUI_SRC := imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/imgui_impl_glfw.cpp imgui/imgui_impl_opengl3.cpp

# Object files
CUDA_OBJ := sph_kernels.o
CPP_OBJ  := main.o
IMGUI_OBJ := $(IMGUI_SRC:.cpp=.o)

# Rules
all: $(TARGET)

$(TARGET): $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(CPP_OBJ): $(CPP_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

imgui/%.o: imgui/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o imgui/*.o

.PHONY: all clean