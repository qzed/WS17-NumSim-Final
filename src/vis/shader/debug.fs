//! Fragment shader for debugging purposes.

#version 330 core

in  vec2 v_texcoord;
out vec4 f_color;

uniform sampler2D u_tex_data;

uniform float u_norm_min = 0.0;
uniform float u_norm_max = 1.4;


vec3 cubehelix(float start, float rotations, float hue, float gamma, float value);

float normscale(float val) {
    return (val - u_norm_min) / (u_norm_max - u_norm_min);
}

void main() {
    float datapoint = normscale(texture(u_tex_data, v_texcoord).r);
    f_color = vec4(cubehelix(0.5, -1.5, 1.0, 1.0, datapoint), 1.0);
}
