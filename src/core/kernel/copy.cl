

//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


__kernel void copy_buf_to_img(
    __write_only image2d_t output,
    __global const float* data
) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int size_x = get_global_size(0);

    const float val = data[INDEX(pos.x, pos.y, size_x)];
    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}
