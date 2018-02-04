//! Kernels for visualization.
//!
//! Kernels to write specific data to the OpenGL target texture.


#define BC_MASK_SELF                    0b00001111

#define BC_SELF_FLUID                   0b0000
#define BC_SELF_NOSLIP                  0b1100
#define BC_SELF_INFLOW                  0b1101
#define BC_SELF_INFLOW_H                0b0101
#define BC_SELF_INFLOW_V                0b1001
#define BC_SELF_SLIP_H                  0b0110
#define BC_SELF_SLIP_V                  0b1010
#define BC_SELF_OUTFLOW                 0b1110


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


//! Write the specified boundary types to the output image.
kernel void visualize_boundaries(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant uchar* b                 // (n + 2) * (m + 2)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int size_x = get_global_size(0);

    const uchar b_self = b[INDEX(pos.x, pos.y, size_x)] & BC_MASK_SELF;
    float val = 0.0;

    switch (b_self) {
    case BC_SELF_FLUID:    val = 0.0 / 7.0; break;
    case BC_SELF_INFLOW:   val = 1.0 / 7.0; break;
    case BC_SELF_INFLOW_H: val = 2.0 / 7.0; break;
    case BC_SELF_INFLOW_V: val = 3.0 / 7.0; break;
    case BC_SELF_OUTFLOW:  val = 4.0 / 7.0; break;
    case BC_SELF_SLIP_H:   val = 5.0 / 7.0; break;
    case BC_SELF_SLIP_V:   val = 6.0 / 7.0; break;
    case BC_SELF_NOSLIP:   val = 7.0 / 7.0; break;
    }

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}


kernel void visualize_p(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* p                 // (n + 2) * (m + 2)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int size_x = get_global_size(0);

    const float val = p[INDEX(pos.x, pos.y, size_x)];
    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}

kernel void visualize_rhs(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* rhs               // n * m
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 size = (int2)(get_global_size(0), get_global_size(1));

    float val = 0.0;
    if ((pos.x > 0) && (pos.y > 0) && (pos.x < (size.x - 1)) && (pos.y < (size.y - 1))) {
        val = rhs[INDEX(pos.x - 1, pos.y - 1, size.x - 2)];
    }

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}


kernel void visualize_u(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* u                 // (n + 3) * (m + 2)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int u_size_x = get_global_size(0) + 1;

    const float val = u[INDEX(pos.x + 1, pos.y, u_size_x)];

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}

kernel void visualize_u_center(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* u                 // (n + 3) * (m + 2)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int u_size_x = get_global_size(0) + 1;

    const float u_cell = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    const float u_left = u[INDEX(pos.x, pos.y, u_size_x)];
    const float val = (u_cell + u_left) / 2.0;

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}


kernel void visualize_v(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* v                 // (n + 2) * (m + 3)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int v_size_x = get_global_size(0);

    const float val = v[INDEX(pos.x, pos.y + 1, v_size_x)];

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}

kernel void visualize_v_center(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* v                 // (n + 2) * (m + 3)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int v_size_x = get_global_size(0);

    const float v_cell = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    const float v_down = v[INDEX(pos.x, pos.y, v_size_x)];
    const float val = (v_cell + v_down) / 2.0;

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}


kernel void visualize_uv_abs(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* u,                // (n + 3) * (m + 2)
    __constant float* v                 // (n + 2) * (m + 3)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int v_size_x = get_global_size(0);
    const int u_size_x = v_size_x + 1;

    const float val_u = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    const float val_v = v[INDEX(pos.x, pos.y + 1, u_size_x)];
    const float val = length((float2)(val_u, val_v));

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}

kernel void visualize_uv_abs_center(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __constant float* u,                // (n + 3) * (m + 2)
    __constant float* v                 // (n + 2) * (m + 3)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int v_size_x = get_global_size(0);
    const int u_size_x = v_size_x + 1;

    const float val_u_cell = u[INDEX(pos.x + 1, pos.y, u_size_x)];
    const float val_u_left = u[INDEX(pos.x, pos.y, u_size_x)];
    const float val_u = (val_u_cell + val_u_left) / 2.0;

    const float val_v_cell = v[INDEX(pos.x, pos.y + 1, v_size_x)];
    const float val_v_down = v[INDEX(pos.x, pos.y, v_size_x)];
    const float val_v = (val_v_cell + val_v_down) / 2.0;
    
    const float val = length((float2)(val_u, val_v));

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}
