//! Vertex shader to render a single, screen-covering triangle.
//!
//! Emits vertices for a screen-covering triangle.
//! Render with `glDrawArrays(GL_TRIANGLES, 0, 3)`.

#version 330 core

out vec2 v_texcoord;


void main() {
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;

    v_texcoord = 0.5 * (vec2(x, y) + vec2(1.0, 1.0));
    gl_Position = vec4(x, y, 0.0, 1.0);
}
