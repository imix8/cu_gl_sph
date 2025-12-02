# Compilers
NVCC := nvcc
CXX  := g++

# Output
TARGET := simple_sph

# Flags
CUDA_PATH ?= /usr/local/cuda
NVCCFLAGS := -O3 -std=c++14 -Iexternal/glfw-3.1.2/include -Iexternal/glm-0.9.7.1 -DGLFW_EXPOSE_NATIVE_GLX

CXXFLAGS  := -O3 -std=c++14 -Wall -I$(CUDA_PATH)/include -I./imgui -Iexternal/glfw-3.1.2/include -Iexternal/glm-0.9.7.1 -DGLFW_EXPOSE_NATIVE_GLX

GLFW_LIB := external/glfw-3.1.2/build/src
GLFW_TARGET := $(GLFW_LIB)/libglfw3.a

# Libraries
LDFLAGS := -lGL -lGLEW -L$(GLFW_LIB) -lglfw3 -lX11 -lXi -lXrandr -lXinerama -lXcursor -lpthread -lcudart

# CPP Source Files
CUDA_SRC := sph_kernels.cu
CPP_SRC  := main.cpp shaders.cpp SimWindow.cpp SimGui.cpp
IMGUI_SRC := imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/imgui_impl_glfw.cpp imgui/imgui_impl_opengl3.cpp

# Object files
CUDA_OBJ := sph_kernels.o
CPP_OBJ  := main.o shaders.o SimWindow.o SimGui.o
IMGUI_OBJ := $(IMGUI_SRC:.cpp=.o)

# Rules
all: $(TARGET)

setup:
	# Create imgui directory
	mkdir -p imgui

	# Download Core ImGui files
	wget -O imgui/imgui.h 			https://raw.githubusercontent.com/ocornut/imgui/master/imgui.h
	wget -O imgui/imgui.cpp 		https://raw.githubusercontent.com/ocornut/imgui/master/imgui.cpp
	wget -O imgui/imgui_draw.cpp 	https://raw.githubusercontent.com/ocornut/imgui/master/imgui_draw.cpp
	wget -O imgui/imgui_tables.cpp 	https://raw.githubusercontent.com/ocornut/imgui/master/imgui_tables.cpp
	wget -O imgui/imgui_widgets.cpp	https://raw.githubusercontent.com/ocornut/imgui/master/imgui_widgets.cpp
	wget -O imgui/imconfig.h		https://raw.githubusercontent.com/ocornut/imgui/master/imconfig.h
	wget -O imgui/imgui_internal.h	https://raw.githubusercontent.com/ocornut/imgui/master/imgui_internal.h
	wget -O imgui/imstb_rectpack.h	https://raw.githubusercontent.com/ocornut/imgui/master/imstb_rectpack.h
	wget -O imgui/imstb_textedit.h	https://raw.githubusercontent.com/ocornut/imgui/master/imstb_textedit.h
	wget -O imgui/imstb_truetype.h	https://raw.githubusercontent.com/ocornut/imgui/master/imstb_truetype.h

	# Download Backends (GLFW and OpenGL3)
	wget -O imgui/imgui_impl_glfw.h				https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h
	wget -O imgui/imgui_impl_glfw.cpp			https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp
	wget -O imgui/imgui_impl_opengl3.h			https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h
	wget -O imgui/imgui_impl_opengl3.cpp		https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp
	wget -O imgui/imgui_impl_opengl3_loader.h 	https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h

# Build GLFW if not already built
$(GLFW_TARGET):
	@echo "Building GLFW..."
	@mkdir -p external/glfw-3.1.2/build
	@cd external/glfw-3.1.2/build && cmake .. && $(MAKE)

$(TARGET): $(GLFW_TARGET) $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ) $(LDFLAGS)
# $(TARGET): $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ)
# 	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Generic rule for .cpp files (handles main.cpp and shaders.cpp)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

imgui/%.o: imgui/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f 	$(TARGET) \
			*.o \
			imgui/*.o \
			imgui.ini

run:
	./simple_sph

.PHONY: all clean