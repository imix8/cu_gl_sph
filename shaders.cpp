/*
    Authors: Ivan Mix, Jacob Dudik, Abhinav Vemulapalli, Nikola Rogers
    Class: ECE6122
    Last Date Modified: 12/2/25
    Description: Shader source code for loading and coloring objects in the simulator
*/

#include "shaders.h"

#include <iostream>
#include <vector>

// ---------------- SHADER SOURCES ----------------
// Vertex shader source code
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;      
layout (location = 1) in vec4 aInstance; 
uniform mat4 view;
uniform mat4 projection;
uniform float radius;
out float vMag; 
void main() {
    vec3 worldPos = aPos * radius + aInstance.xyz; 
    glPosition = projection * view * vec4(worldPos, 1.0);
    vMag = aInstance.w;
}
)";

// Fragment shader source code to color particles based on their velocity for either blue or a "plasma"
const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in float vMag;
uniform float vMin;
uniform float vMax;
uniform int colorMode; // 0 = Plasma, 1 = Blue

// Plasma Color Scheme
vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.05, 0.03, 0.53);
    vec3 c1 = vec3(0.50, 0.03, 0.62);
    vec3 c2 = vec3(0.83, 0.23, 0.48);
    vec3 c3 = vec3(0.98, 0.52, 0.19);
    vec3 c4 = vec3(0.94, 0.98, 0.13);

    if (t < 0.7) return mix(c0, c1, t * 4.0);
    else if (t < 0.8) return mix(c1, c2, (t - 0.7) * 4.0);
    else if (t < 0.9) return mix(c2, c3, (t - 0.8) * 4.0);
    else return mix(c3, c4, (t - 0.9) * 4.0);
}

// Blue Color Scheme
vec3 ocean(float t) {
    t = clamp(t, 0.0, 1.0);
    // Dark Navy -> Azure -> White (Foam)
    vec3 c0 = vec3(0.0, 0.05, 0.2);   // Deep/Slow
    vec3 c1 = vec3(0.0, 0.3, 0.7);    // Mid
    vec3 c2 = vec3(0.0, 0.8, 1.0);    // Fast
    vec3 c3 = vec3(1.0, 1.0, 1.0);    // Very Fast (Foam)

    if (t < 0.5) return mix(c0, c1, t * 3.0);
    else if (t < 0.75) return mix(c1, c2, (t - 0.5) * 3.0);
    else return mix(c2, c3, (t - 0.75) * 3.0);
}

void main() {
    float t = (vMag - vMin) / (vMax - vMin + 0.00001);
    
    vec3 col;
    if (colorMode == 1) {
        col = ocean(t);
    } else {
        col = plasma(t);
    }
    
    FragColor = vec4(col, 1.0);
}
)";

// Compile the specified shader code
GLuint compileShader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "SHADER ERROR ("
                  << (type == GL_VERTEX_SHADER ? "VERT" : "FRAG") << "):\n"
                  << infoLog << std::endl;
    }
    return shader;
}

// Compile and link all shader code needed for the program
GLuint createShaderProgram()
{
    GLuint program = glCreateProgram();
    GLuint v = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    glAttachShader(program, v);
    glAttachShader(program, f);
    glLinkProgram(program);

    // Cleanup shaders as they are linked into the program now
    glDeleteShader(v);
    glDeleteShader(f);

    return program;
}