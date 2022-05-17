#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 1.0, 0.0);

layout(set = 0, binding = 0) uniform Data {
    mat4 world_view;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;
layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));

    vec3 regular_color = texture(tex, tex_coords).rgb + vec3(0.0);
    vec3 dark_color = regular_color / 4.0;

    f_color = vec4(mix(regular_color, dark_color, brightness), 1.0);
}
