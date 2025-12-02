# -----------------------------------------------------------------------------
# HYBRID CROSS-PLATFORM MAKEFILE
# -----------------------------------------------------------------------------

APP_NAME := simple_sph

# -----------------------------------------------------------------------------
# 1. OS DETECTION & COMPILER SETUP
# -----------------------------------------------------------------------------

ifeq ($(OS),Windows_NT)
    # --- WINDOWS (MSVC + NVCC) ---
    TARGET_OS := Windows
    TARGET    := $(APP_NAME).exe
    OBJ_EXT   := obj
    
    # Compilers
    NVCC := nvcc
    CXX  := cl
    
    # Paths
    ifndef CUDA_HOME
        CUDA_HOME := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
    endif
    
    GLEW_PATH := external/glew-2.1.0
    GLFW_PATH := external/glfw-3.1.2
    
    # OUTPUT LOCATION FOR GLFW
    GLFW_BUILD_DIR := $(GLFW_PATH)/build_msvc
    GLFW_LIB_DIR   := $(GLFW_BUILD_DIR)/src
    GLFW_LIB       := $(GLFW_LIB_DIR)/glfw3.lib
    
    # Includes
    INCLUDES := -I"$(CUDA_HOME)/include" -I./imgui -I$(GLFW_PATH)/include -I$(GLEW_PATH)/include -Iexternal/glm-0.9.7.1
    
    # Flags for MSVC (cl.exe)
    # Use hyphens (-) for flags to avoid path interpretation issues
    CXXFLAGS := -O2 -std:c++14 -EHsc -MD -nologo $(INCLUDES) \
                -DGLEW_STATIC -DGLFW_EXPOSE_NATIVE_WIN32 -DGLFW_EXPOSE_NATIVE_WGL
    
    # Flags for NVCC
    # Use -MD to match MSVC runtime
    NVCCFLAGS_WIN := -Xcompiler -MD
                
    # Libraries
    LDFLAGS := -L$(GLFW_LIB_DIR) \
               -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 -lcudart_static
    
    COMPILE_CPP = $(CXX) $(CXXFLAGS) -c $< -Fo$@
    
    # GLFW BUILD COMMAND (Windows)
    # FIX: Use forward slashes and && for compatibility with Make's shell (Git Bash/MinGW)
    # Added -DCMAKE_BUILD_TYPE=Release to fix the MSVCRTD warning
    BUILD_GLFW_CMD = mkdir -p $(GLFW_BUILD_DIR) && \
                     cd $(GLFW_BUILD_DIR) && \
                     cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release .. && \
                     nmake

else
    # --- LINUX (GCC + NVCC) ---
    TARGET_OS := Linux
    TARGET    := $(APP_NAME)
    OBJ_EXT   := o
    
    NVCC := nvcc
    CXX  := g++
    
    ifndef CUDA_HOME
        CUDA_HOME := /usr/local/cuda
    endif

    GLFW_PATH := external/glfw-3.1.2
    
    GLFW_BUILD_DIR := $(GLFW_PATH)/build
    GLFW_LIB_DIR   := $(GLFW_BUILD_DIR)/src
    GLFW_LIB       := $(GLFW_LIB_DIR)/libglfw3.a
    
    INCLUDES := -I$(CUDA_HOME)/include -I./imgui -I$(GLFW_PATH)/include -Iexternal/glm-0.9.7.1
    
    CXXFLAGS := -O3 -std=c++14 -Wall $(INCLUDES) -DGLFW_EXPOSE_NATIVE_X11 -DGLFW_EXPOSE_NATIVE_GLX -DGLEW_STATIC
    NVCCFLAGS_WIN := 
    
    LDFLAGS := -lGL -lGLEW -lglfw -lX11 -lXi -lXrandr -lXinerama -lXcursor -lpthread -lcudart
    
    COMPILE_CPP = $(CXX) $(CXXFLAGS) -c $< -o $@
    
    BUILD_GLFW_CMD = mkdir -p $(GLFW_BUILD_DIR) && \
                     cd $(GLFW_BUILD_DIR) && \
                     cmake -DCMAKE_BUILD_TYPE=Release .. && \
                     make
endif

# -----------------------------------------------------------------------------
# 2. FILES & OBJECTS
# -----------------------------------------------------------------------------

CUDA_SRC  := sph_kernels.cu
CPP_SRC   := main.cpp shaders.cpp SimWindow.cpp SimGui.cpp external/glew-2.1.0/src/glew.c
IMGUI_SRC := imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp \
             imgui/imgui_widgets.cpp imgui/imgui_impl_glfw.cpp imgui/imgui_impl_opengl3.cpp

CUDA_OBJ  := $(CUDA_SRC:.cu=.$(OBJ_EXT))
CPP_OBJ   := $(CPP_SRC:.cpp=.$(OBJ_EXT))
IMGUI_OBJ := $(IMGUI_SRC:.cpp=.$(OBJ_EXT))

# NVCC Flags
NVCCFLAGS := -O3 -std=c++14 $(INCLUDES) $(NVCCFLAGS_WIN)

# -----------------------------------------------------------------------------
# 3. BUILD RULES
# -----------------------------------------------------------------------------

all: info $(TARGET)

info:
	@echo "------------------------------------"
	@echo "OS       : $(TARGET_OS)"
	@echo "Target   : $(TARGET)"
	@echo "GLFW Lib : $(GLFW_LIB)"
	@echo "------------------------------------"

$(TARGET): $(GLFW_LIB) $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CPP_OBJ) $(CUDA_OBJ) $(IMGUI_OBJ) $(LDFLAGS)

$(GLFW_LIB):
	@echo "--- Building GLFW Library ---"
	$(BUILD_GLFW_CMD)
	@echo "--- GLFW Build Complete ---"

%.$(OBJ_EXT): %.cpp
	$(COMPILE_CPP)

%.$(OBJ_EXT): %.c
	$(COMPILE_CPP)

%.$(OBJ_EXT): %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

imgui/%.$(OBJ_EXT): imgui/%.cpp
	$(COMPILE_CPP)

clean:
	rm -f $(TARGET) *.$(OBJ_EXT) imgui/*.$(OBJ_EXT) imgui.ini
	rm -rf external/glew-2.1.0/build
	rm -rf external/glfw-3.1.2/build
	rm -rf external/glfw-3.1.2/build_msvc
	rm -rf external/glm-0.9.7.1/build
	rm -f *.lib
	rm -f *.exp

run: all
	./$(TARGET)

.PHONY: all clean run info