
__kernel void zero_float(__write_only __global float* buf) {
    buf[get_global_id(0)] = 0.0;
}
