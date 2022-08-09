#version 450
layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
    vec4 tex_color = texture(tex, tex_coords);
    if (tex_color.a < 0.01)
        discard;

    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
