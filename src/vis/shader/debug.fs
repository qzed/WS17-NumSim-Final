//! Fragment shader for debugging purposes.

#version 330 core

in  vec2 v_texcoord;
out vec4 f_color;

uniform sampler2D u_tex_data;


void main() {
    vec2 d = texture(u_tex_data, v_texcoord).rg;
    f_color = vec4(d.rg, 0.0, 1.0);
}
