//! Kernel for the right-hand side of the pressure equation.
//!


#define BC_MASK_SELF                    0b00001111
#define BC_SELF_FLUID                   0b0000


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


//! Computes the right-hand-side of the equation to be solved.
//! 
//! Grid sizes:
//! - f: (n + 3) * (m + 2)      i.e. has boundaries and is staggered in x direction 
//! - g: (n + 2) * (m + 3)      i.e. has boundaries and is staggered in y direction
//! - b: (n + 2) * (m + 2)      i.e. has boundaries but is not staggered
//! - rhs: n * m                i.e. has no boundaries and is not staggered
//! where n * m is the size of the interior.
//!
__kernel void compute_rhs(
    __read_only __global float* f,
    __read_only __global float* g,
    __write_only __global float* rhs,
    __read_only __global uchar* b,
    __read_only float dt,
    __read_only float2 h
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n, len.y = m

    const int rhs_size_x = get_global_size(0);
    const int b_size_x = rhs_size_x + 2;
    const int f_size_x = rhs_size_x + 3;
    const int g_size_x = rhs_size_x + 2;

    // only execute on fluid cells
    const uchar b_center = b[INDEX(pos.x + 1, pos.y + 1, b_size_x)];
    if ((b_center & BC_MASK_SELF) != BC_SELF_FLUID) {
        return;
    }

    // load f
    const float f_center = f[INDEX(pos.x + 2, pos.y + 1, f_size_x)];
    const float f_left   = f[INDEX(pos.x + 1, pos.y + 1, f_size_x)];

    // load g
    const float g_center = g[INDEX(pos.x + 1, pos.y + 2, g_size_x)];
    const float g_down   = g[INDEX(pos.x + 1, pos.y + 1, g_size_x)];

    // f_dx_l, g_dy_l 
    const float f_dx_l = (f_center - f_left) / h.x;
    const float g_dy_l = (g_center - g_down) / h.y;

    // store result
    rhs[INDEX(pos.x, pos.y, rhs_size_x)] = (f_dx_l + g_dy_l) / dt;
}
