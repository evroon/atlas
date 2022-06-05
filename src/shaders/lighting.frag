#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_position;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 3) uniform LightingData {
    vec4 ambient_color;
    vec4 directional_direction;
    vec4 directional_color;
    int preview_type;
} u_lighting;

vec3 main_pass(vec3 albedo, vec3 normal, vec3 position) {
    vec3 ambient_color = u_lighting.ambient_color.a * u_lighting.ambient_color.rgb;
    float directional_intensity = max(dot(normal, u_lighting.directional_direction.xyz), 0.0);
    vec3 directional_color = directional_intensity * u_lighting.directional_color.xyz;
    return (ambient_color + directional_color) * albedo;
}

void main() {
    vec3 albedo = subpassLoad(u_color).rgb;
    vec3 normal = subpassLoad(u_normal).rgb;
    vec3 position = subpassLoad(u_position).rgb;

    vec3 final_output = 0.0.xxx;

    if (u_lighting.preview_type == 1) {
        final_output = albedo;
    } else if (u_lighting.preview_type == 2) {
        final_output = normal;
    } else if (u_lighting.preview_type == 3) {
        final_output = position;
    } else {
        final_output = main_pass(albedo, normal, position);
    }

    f_color = vec4(final_output, 1.0);
}
