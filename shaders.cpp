#include "shaders.h"

#include <iostream>
#include <vector>

// ---------------- SHADER SOURCES ----------------

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;      
layout (location = 1) in vec4 aInstance; 
uniform mat4 view;
uniform mat4 projection;
uniform float radius;
out float vMag; 
void main() {
    vec3 worldPos = aPos * radius + aInstance.xyz; 
    gl_Position = projection * view * vec4(worldPos, 1.0);
    vMag = aInstance.w;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in float vMag;
uniform float vmin;
uniform float vmax;

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.05, 0.03, 0.53);
    vec3 c1 = vec3(0.50, 0.03, 0.62);
    vec3 c2 = vec3(0.83, 0.23, 0.48);
    vec3 c3 = vec3(0.98, 0.52, 0.19);
    vec3 c4 = vec3(0.94, 0.98, 0.13);

    if (t < 0.25) return mix(c0, c1, t * 4.0);
    else if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
    else if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    else return mix(c3, c4, (t - 0.75) * 4.0);
}

void main() {
    float t = (vMag - vmin) / (vmax - vmin + 0.00001);
    vec3 col = plasma(t);
    FragColor = vec4(col, 1.0);
}
)";

// ---------------- INTERNAL HELPER ----------------

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "SHADER ERROR ("
                  << (type == GL_VERTEX_SHADER ? "VERT" : "FRAG") << "):\n"
                  << infoLog << std::endl;
    }
    return shader;
}

// ---------------- PUBLIC FUNCTION ----------------

GLuint createShaderProgram() {
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