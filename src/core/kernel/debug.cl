
kernel void write_image(write_only image2d_t output) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const float2 size = {get_global_size(0), get_global_size(1)};

    float4 result = {pos.x / size.x, pos.y / size.y, 1.0, 1.0};

    write_imagef(output, pos, result);
}
