#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 albedo = subpassLoad(u_color).rgb;
    vec3 normal = subpassLoad(u_normals).rgb;
    f_color = vec4(albedo, 1.0);
}
