
__kernel void write_image(__write_only image2d_t output) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const float2 size = {get_global_size(0), get_global_size(1)};

    float2 fpos = {pos.x / size.x, pos.y / size.y};
    float val = length(fpos);

    write_imagef(output, pos, (float4)(val, 0.0, 0.0, 0.0));
}
