//! Kernels for the momentum equation.
//!


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


//! Computes the momentum equation for F.
//!
//! Grid sizes:
//! - u, f: (n + 3) * (m + 2)   i.e. has boundaries and is staggered in x direction 
//! - v: (n + 2) * (m + 3)      i.e. has boundaries and is staggered in y direction
//! where n * m is the size of the interior.
//!
__kernel void momentum_eq_f(
    __read_only __global float* u,
    __read_only __global float* v,
    __write_only __global float* f,
    __read_only int u_size_x,
    __read_only int v_size_x,
    __read_only float alpha,
    __read_only float re,
    __read_only float dt,
    __read_only float2 h
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x > 0 && pos.x < (len.x - 1) && pos.y > 0 && pos.y < (len.y - 1)
    // with len.x = n + 3, len.y = m + 2

    // TODO: only execute on fluid-to-fluid cell boundaries (?)

    // load u
    float u_center     = u[INDEX(pos.x, pos.y, u_size_x)];
    float u_left       = u[INDEX(pos.x - 1, pos.y, u_size_x)];
    float u_right      = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    float u_top        = u[INDEX(pos.x, pos.y + 1, u_size_x)];
    float u_down       = u[INDEX(pos.x, pos.y - 1, u_size_x)];

    // load v
    float v_center     = v[INDEX(pos.x - 1, pos.y + 1, v_size_x)];
    float v_right      = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    float v_down       = v[INDEX(pos.x - 1, pos.y, v_size_x)];
    float v_down_right = v[INDEX(pos.x, pos.y, v_size_x)];

    // dxx(u), dyy(u)
    float dxx = (u_right - 2.0 * u_center + u_left) / (h.x * h.x);
    float dyy = (u_top - 2.0 * u_center + u_down) / (h.y * h.y);
    float acc = (dxx + dyy) / re;

    // dc_udu_x
    float dc_udu_x_ar = (u_center + u_right) / 2.0;
    float dc_udu_x_al = (u_left + u_center) / 2.0;
    float dc_udu_a = (dc_udu_x_ar * dc_udu_x_ar) - (dc_udu_x_al * dc_udu_x_al);

    float dc_udu_x_br = (abs(u_center + u_right) / 2.0) * ((u_center - u_right) / 2.0);
    float dc_udu_x_bl = (abs(u_left + u_center) / 2.0) * ((u_left - u_center) / 2.0);
    float dc_udu_x_b = dc_udu_x_br - dc_udu_x_bl;

    float dc_udu_x = (dc_udu_x_a / h.x) + (alpha / h.x) * dc_udu_x_b;
    acc -= dc_udu_x;

    // dc_vdu_y
    float dc_vdu_y_ar = ((v_center + v_right) / 2.0) * ((u_center + u_top) / 2.0);
    float dc_vdu_y_al = ((v_down + v_down_right) / 2.0) * ((u_down + u_center) / 2.0);
    float dc_vdu_y_a = dc_vdu_y_ar - dc_vdu_y_al;

    float dc_vdu_y_br = (abs(v_center + v_right) / 2.0) * ((u_center - u_top) / 2.0);
    float dc_vdu_y_bl = (abs(v_down + v_down_right) / 2.0) * ((u_down - u_center) / 2.0);
    float dc_vdu_y_b = dc_vdu_y_br - dc_vdu_y_bl;

    float dc_vdu_y = (dc_vdu_y_a / h.y) + (alpha / h.y) * dc_vdu_y_b;
    acc -= dc_vdu_y;

    // store result
    f[INDEX(pos.x, pos.y, u_size_x)] = u_center + dt * acc;
}


//! Computes the momentum equation for G.
//!
//! Grid sizes:
//! - u: (n + 3) * (m + 2)      i.e. has boundaries and is staggered in x direction 
//! - v, g: (n + 2) * (m + 3)   i.e. has boundaries and is staggered in y direction
//! where n * m is the size of the interior.
//!
__kernel void momentum_eq_g(
    __read_only __global float* u,
    __read_only __global float* v,
    __write_only __global float* g,
    __read_only int u_size_x,
    __read_only int v_size_x,
    __read_only float alpha,
    __read_only float re,
    __read_only float dt,
    __read_only float2 h
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x > 0 && pos.x < (len.x - 1) && pos.y > 0 && pos.y < (len.y - 1)
    // with len.x = n + 2, len.y = m + 3

    // TODO: only execute on fluid-to-fluid cell boundaries

    // load v
    float v_center   = v[INDEX(pos.x, pos.y, v_size_x)];
    float v_left     = v[INDEX(pos.x - 1, pos.y, v_size_x)];
    float v_right    = v[INDEX(pos.x + 1, pos.y, v_size_x)];
    float v_top      = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    float v_down     = v[INDEX(pos.x, pos.y - 1, v_size_x)];

    // load u
    float u_center   = u[INDEX(pos.x + 1, pos.y - 1, u_size_x)];
    float u_top      = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    float u_left     = u[INDEX(pos.x, pos.y - 1, u_size_x)];
    float u_top_left = u[INDEX(pos.x, pos.y, u_size_x)];

    // dxx(v), dyy(v)
    float dxx = (v_right - 2.0 * v_center + v_left) / (h.x * h.x);
    float dyy = (v_top - 2.0 * v_center + v_down) / (h.y * h.y);
    float acc = (dxx + dyy) / re;

    // dc_vdv_y
    float dc_vdv_y_at = (v_center + v_top) / 2.0;
    float dc_vdv_y_ad = (v_down + v_center) / 2.0;
    float dc_vdv_y_a = (dc_vdv_y_at * dc_vdv_y_at) - (dc_vdv_y_ad * dc_vdv_y_ad);

    float dc_vdv_y_bt = (abs(v_center + v_top) / 2.0) * ((v_center - v_top) / 2.0);
    float dc_vdv_y_bd = (abs(v_down + v_center) / 2.0) * ((v_down - v_center) / 2.0);
    float dc_vdv_y_b = dc_vdv_y_bt - dc_vdv_y_bd;

    float dc_vdv_y = (dc_vdv_y_a / h.y) + (alpha / h.y) * dc_vdv_y_b;
    acc -= dc_vdv_y;

    // dc_udv_x
    float dc_udv_x_ar = ((v_center + v_right) / 2.0) * ((u_center + u_top) / 2.0);
    float dc_udv_x_al = ((v_left + v_center) / 2.0) * ((u_left + u_top_left) / 2.0);
    float dc_udv_x_a = dc_udv_x_ar - dc_udv_x_al;

    float dc_udv_x_br = ((v_center - v_right) / 2.0) * (abs(u_center + u_top) / 2.0);
    float dc_udv_x_bl = ((v_left - v_center) / 2.0) * (abs(u_left + u_top_left) / 2.0);
    float dc_udv_x_b = dc_udv_x_br - dc_udv_x_bl;

    float dc_udv_x = (dc_udv_x_a / h.x) + (alpha / h.x) * dc_udv_x_b;
    acc -= dc_udv_x;

    // store result
    g[INDEX(pos.x, pos.y, v_size_x)] = v_center + dt * acc;
}
