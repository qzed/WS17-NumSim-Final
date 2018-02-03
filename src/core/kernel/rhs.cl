//! Kernel for the right-hand side of the pressure equation.
//!


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


//! Computes the right-hand-side of the equation to be solved.
//! 
//! Grid sizes:
//! - f: (n + 3) * (m + 2)      i.e. has boundaries and is staggered in x direction 
//! - g: (n + 2) * (m + 3)      i.e. has boundaries and is staggered in y direction
//! - result: n * m             i.e. has no boundaries and is not staggered
//! where n * m is the size of the interior.
//!
__kernel void rhs(
    __read_only __global float* f,
    __read_only __global float* g,
    __write_only __global float* result,
    __read_only int f_size_x,
    __read_only int g_size_x,
    __read_only int result_size_x,
    __read_only float dt,
    __read_only float2 h
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n, len.y = m

    // TODO: only execute on fluid cells

    // load f
    float f_center = f[INDEX(pos.x + 2, pos.y + 1, f_size_x)];
    float f_left   = f[INDEX(pos.x + 1, pos.y + 1, f_size_x)];

    // load g
    float g_center = g[INDEX(pos.x + 1, pos.y + 2, g_size_x)];
    float g_down   = g[INDEX(pos.x + 1, pos.y + 1; g_size_x)];

    // f_dx_l, g_dy_l 
    float f_dx_l = (f_center - f_left) / h.x;
    float g_dy_l = (g_center - g_down) / h.y;

    // store result
    result[INDEX(pos.x, pos.y, result_size_x)] = (f_dx_l + g_dy_l) / dt;
}