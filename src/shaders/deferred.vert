#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 out_coords;

layout(set = 0, binding = 0) uniform Data {
    mat4 world_view;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    out_coords = tex_coord;
    v_normal = mat3(uniforms.world) * normal;
    gl_Position = uniforms.proj * uniforms.world_view * vec4(position, 1.0);
}
