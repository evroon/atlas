#version 450
layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;

layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
    vec4 tex_color = texture(tex, tex_coords);
    if (tex_color.a < 0.01)
        discard;

    vec3 regular_color = tex_color.rgb + vec3(0.0);
    f_color = vec4(regular_color, 1.0);
    f_normal = vec4(in_normal, 1.0);
}
