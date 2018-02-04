//! Kernel to calculate the final velocities.
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


//! Computes the new velocities based on the pressure and preliminary velocities.
//! 
//! Grid sizes:
//! - f, u: (n + 3) * (m + 2)      i.e. has boundaries and is staggered in x direction 
//! - g, v: (n + 2) * (m + 3)      i.e. has boundaries and is staggered in y direction
//! - p: (n + 1) * (m + 1)         i.e. has boundaries but is not staggered
//! where n * m is the size of the interior.
//!
__kernel void new_velocities(
    __global float* p,
    __global float* f,
    __global float* g,
    __global float* u,
    __global float* v,
    __global uchar* b,
    const float dt,
    const float2 h
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n + 2, len.y = m + 2

    const int p_size_x = get_global_size(0);
    const int u_size_x = p_size_x + 1;
    const int v_size_x = p_size_x;

    // get cell type
    const uchar b_cell = b[INDEX(pos.x, pos.y, p_size_x)];

    // load common
    const float p_center = p[INDEX(pos.x, pos.y, p_size_x)];

    // calculate u (only for cell-boundaries inside fluid
    if ((b_cell & BC_MASK_SELF) == BC_SELF_FLUID && BC_IS_NEIGHBOR_RIGHT_FLUID(b_cell)) {
        const float p_right = p[INDEX(pos.x + 1, pos.y, p_size_x)];
        const float f_center = f[INDEX(pos.x + 1, pos.y, u_size_x)];

        const float p_dx_r = (p_right - p_center) / h.x;

        u[INDEX(pos.x + 1, pos.y, u_size_x)] = f_center - dt * p_dx_r;
    }

    // calculate v (only for cell-boundaries inside fluid
    if ((b_cell & BC_MASK_SELF) == BC_SELF_FLUID && BC_IS_NEIGHBOR_TOP_FLUID(b_cell)) {
        const float p_top = p[INDEX(pos.x, pos.y + 1, p_size_x)];
        const float g_center = g[INDEX(pos.x, pos.y + 1, v_size_x)];

        const float p_dy_r = (p_top - p_center) / h.y;

        v[INDEX(pos.x, pos.y + 1, v_size_x)] = g_center - dt * p_dy_r;
    }
}
