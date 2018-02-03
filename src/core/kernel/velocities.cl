//! Kernel to calculate the final velocities.
//!


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
    __read_only __global float* p,
    __read_only __global float* f,
    __read_only __global float* g,
    __write_only __global float* u,
    __write_only __global float* v,
    __read_only int p_size_x,
    __read_only int u_size_x,
    __read_only int v_size_x,
    __read_only float dt,
    __read_only float2 h
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n + 2, len.y = m + 2

    // load common
    float p_center = p[INDEX(pos.x, pos.y, p_size_x)];

    // calculate u
    {   // TODO: only for boundaries to/between fluid(-like) cells on u (inc. inflow/pressure/outflow/...) (?)
        float p_right = p[INDEX(pos.x + 1, pos.y, p_size_x)];
        float p_dx_r = (p_right - p_center) / h.x;
        u[INDEX(pos.x + 1, pos.y, u_size_x)] = f_center - dt * p_dx_r;
    }

    // calculate v
    {   // TODO: only for boundaries to/between fluid(-like) cells on v (inc. inflow/pressure/outflow/...) (?)
        float p_top = p[INDEX(pos.x, pos.y + 1, p_size_x)];
        float p_dy_r = (p_top - p_center) / h.y;
        v[INDEX(pos.x, pos.y + 1, v_size_x)] = g_center - dt * p_dy_r;
    }
}
