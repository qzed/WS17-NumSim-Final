//! Kernels for the momentum equation.
//!

#define BC_MASK_SELF                    0b00001111
#define BC_SELF_FLUID                   0b0000

#define BC_MASK_NEIGHBOR_LEFT           0b10000000
#define BC_MASK_NEIGHBOR_RIGHT          0b01000000
#define BC_MASK_NEIGHBOR_BOTTOM         0b00100000
#define BC_MASK_NEIGHBOR_TOP            0b00010000

#define BC_IS_NEIGHBOR_LEFT_FLUID(x)    ((x) & (BC_MASK_NEIGHBOR_LEFT))
#define BC_IS_NEIGHBOR_RIGHT_FLUID(x)   ((x) & (BC_MASK_NEIGHBOR_RIGHT))
#define BC_IS_NEIGHBOR_BOTTOM_FLUID(x)  ((x) & (BC_MASK_NEIGHBOR_BOTTOM))
#define BC_IS_NEIGHBOR_TOP_FLUID(x)     ((x) & (BC_MASK_NEIGHBOR_TOP))

//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


//! Computes the momentum equation for F.
//!
//! Grid sizes:
//! - u, f: (n + 3) * (m + 2)   i.e. has boundaries and is staggered in x direction 
//! - v: (n + 2) * (m + 3)      i.e. has boundaries and is staggered in y direction
//! - b: (n + 2) * (m + 2)      i.e. has boundaries but is not staggered
//! where n * m is the size of the interior.
//!
__kernel void momentum_eq_f(
    __constant float* u,
    __constant float* v,
    __global float* f,
    __constant uchar* b,
    const float alpha,
    const float re,
    const float dt,
    const float2 h
) {
    const int2 pos = (int2)(get_global_id(0) + 1, get_global_id(1));
    // assumes: pos.x > 0 && pos.x < (len.x - 1) && pos.y > 0 && pos.y < (len.y - 1)
    // with len.x = n + 3, len.y = m + 2

    const int b_size_x = get_global_size(0);
    const int u_size_x = b_size_x + 1;
    const int v_size_x = b_size_x;

    // only execute on fluid-to-fluid cell boundaries
    const uchar b_center = b[INDEX(pos.x - 1, pos.y, b_size_x)];
    if ((b_center & BC_MASK_SELF) != BC_SELF_FLUID || !BC_IS_NEIGHBOR_RIGHT_FLUID(b_center)) {
        return;
    }

    // load u
    const float u_center     = u[INDEX(pos.x, pos.y, u_size_x)];
    const float u_left       = u[INDEX(pos.x - 1, pos.y, u_size_x)];
    const float u_right      = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    const float u_top        = u[INDEX(pos.x, pos.y + 1, u_size_x)];
    const float u_down       = u[INDEX(pos.x, pos.y - 1, u_size_x)];

    // load v
    const float v_center     = v[INDEX(pos.x - 1, pos.y + 1, v_size_x)];
    const float v_right      = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    const float v_down       = v[INDEX(pos.x - 1, pos.y, v_size_x)];
    const float v_down_right = v[INDEX(pos.x, pos.y, v_size_x)];

    // dxx(u), dyy(u)
    const float dxx = (u_right - 2.0 * u_center + u_left) / (h.x * h.x);
    const float dyy = (u_top - 2.0 * u_center + u_down) / (h.y * h.y);
    float acc = (dxx + dyy) / re;

    // dc_udu_x
    const float dc_udu_x_ar = (u_center + u_right) / 2.0;
    const float dc_udu_x_al = (u_left + u_center) / 2.0;
    const float dc_udu_x_a = (dc_udu_x_ar * dc_udu_x_ar) - (dc_udu_x_al * dc_udu_x_al);

    const float dc_udu_x_br = (fabs(u_center + u_right) / 2.0) * ((u_center - u_right) / 2.0);
    const float dc_udu_x_bl = (fabs(u_left + u_center) / 2.0) * ((u_left - u_center) / 2.0);
    const float dc_udu_x_b = dc_udu_x_br - dc_udu_x_bl;

    const float dc_udu_x = (dc_udu_x_a / h.x) + (alpha / h.x) * dc_udu_x_b;
    acc -= dc_udu_x;

    // dc_vdu_y
    const float dc_vdu_y_ar = ((v_center + v_right) / 2.0) * ((u_center + u_top) / 2.0);
    const float dc_vdu_y_al = ((v_down + v_down_right) / 2.0) * ((u_down + u_center) / 2.0);
    const float dc_vdu_y_a = dc_vdu_y_ar - dc_vdu_y_al;

    const float dc_vdu_y_br = (fabs(v_center + v_right) / 2.0) * ((u_center - u_top) / 2.0);
    const float dc_vdu_y_bl = (fabs(v_down + v_down_right) / 2.0) * ((u_down - u_center) / 2.0);
    const float dc_vdu_y_b = dc_vdu_y_br - dc_vdu_y_bl;

    const float dc_vdu_y = (dc_vdu_y_a / h.y) + (alpha / h.y) * dc_vdu_y_b;
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
    __constant float* u,
    __constant float* v,
    __global float* g,
    __constant uchar* b,
    const float alpha,
    const float re,
    const float dt,
    const float2 h
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1) + 1);
    // assumes: pos.x > 0 && pos.x < (len.x - 1) && pos.y > 0 && pos.y < (len.y - 1)
    // with len.x = n + 2, len.y = m + 3

    const int b_size_x = get_global_size(0);
    const int u_size_x = b_size_x + 1;
    const int v_size_x = b_size_x;

    // only execute on fluid-to-fluid cell boundaries
    const uchar b_center = b[INDEX(pos.x, pos.y - 1, b_size_x)];
    if ((b_center & BC_MASK_SELF) != BC_SELF_FLUID || !BC_IS_NEIGHBOR_TOP_FLUID(b_center)) {
        return;
    }

    // load v
    const float v_center   = v[INDEX(pos.x, pos.y, v_size_x)];
    const float v_left     = v[INDEX(pos.x - 1, pos.y, v_size_x)];
    const float v_right    = v[INDEX(pos.x + 1, pos.y, v_size_x)];
    const float v_top      = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    const float v_down     = v[INDEX(pos.x, pos.y - 1, v_size_x)];

    // load u
    const float u_center   = u[INDEX(pos.x + 1, pos.y - 1, u_size_x)];
    const float u_top      = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    const float u_left     = u[INDEX(pos.x, pos.y - 1, u_size_x)];
    const float u_top_left = u[INDEX(pos.x, pos.y, u_size_x)];

    // dxx(v), dyy(v)
    const float dxx = (v_right - 2.0 * v_center + v_left) / (h.x * h.x);
    const float dyy = (v_top - 2.0 * v_center + v_down) / (h.y * h.y);
    float acc = (dxx + dyy) / re;

    // dc_vdv_y
    const float dc_vdv_y_at = (v_center + v_top) / 2.0;
    const float dc_vdv_y_ad = (v_down + v_center) / 2.0;
    const float dc_vdv_y_a = (dc_vdv_y_at * dc_vdv_y_at) - (dc_vdv_y_ad * dc_vdv_y_ad);

    const float dc_vdv_y_bt = (fabs(v_center + v_top) / 2.0) * ((v_center - v_top) / 2.0);
    const float dc_vdv_y_bd = (fabs(v_down + v_center) / 2.0) * ((v_down - v_center) / 2.0);
    const float dc_vdv_y_b = dc_vdv_y_bt - dc_vdv_y_bd;

    float dc_vdv_y = (dc_vdv_y_a / h.y) + (alpha / h.y) * dc_vdv_y_b;
    acc -= dc_vdv_y;

    // dc_udv_x
    const float dc_udv_x_ar = ((v_center + v_right) / 2.0) * ((u_center + u_top) / 2.0);
    const float dc_udv_x_al = ((v_left + v_center) / 2.0) * ((u_left + u_top_left) / 2.0);
    const float dc_udv_x_a = dc_udv_x_ar - dc_udv_x_al;

    const float dc_udv_x_br = ((v_center - v_right) / 2.0) * (fabs(u_center + u_top) / 2.0);
    const float dc_udv_x_bl = ((v_left - v_center) / 2.0) * (fabs(u_left + u_top_left) / 2.0);
    const float dc_udv_x_b = dc_udv_x_br - dc_udv_x_bl;

    const float dc_udv_x = (dc_udv_x_a / h.x) + (alpha / h.x) * dc_udv_x_b;
    acc -= dc_udv_x;

    // store result
    g[INDEX(pos.x, pos.y, b_size_x)] = v_center + dt * acc;
}
