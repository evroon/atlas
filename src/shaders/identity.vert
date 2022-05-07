#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world_view;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    v_normal = transpose(inverse(mat3(uniforms.world_view))) * normal;
    gl_Position = uniforms.proj * uniforms.world_view * vec4(position, 1.0);
}
