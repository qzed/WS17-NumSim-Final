//! Visualize object- and domain-boundaries.
//!


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
kernel void write_image(
    __write_only image2d_t output,      // (n + 2) * (m + 2)
    __read_only __global uchar b,       // (n + 2) * (m + 2)
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int size_x = get_global_size(0);

    const uchar b_self = b[INDEX(pos.x, pos.y, size.x)] & BC_MASK_SELF;
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
