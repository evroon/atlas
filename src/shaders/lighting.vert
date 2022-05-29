#version 450

layout(location = 0) in vec2 position;

layout(set = 0, binding = 2) uniform Data {
    mat4 world_view;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 a = uniforms.proj * uniforms.world_view;
    gl_Position = vec4(position, 0.0, 1.0);
}
