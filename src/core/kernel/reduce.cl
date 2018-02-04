//! Multi-Stage Reductions for floating-point values.
//!
//! Based on https://www.cl.cam.ac.uk/teaching/1617/AdvGraph/07_OpenCL.pdf
//!


__kernel void reduce_max_abs(
    __global float const* input,
    __global float* output,
    __local float* shared,
    const uint n
) {
    const int global_idx = get_global_id(0);
    const int global_len = get_global_size(0);
    const int local_idx = get_local_id(0);
    const int local_len = get_local_size(0);

    // Initialize accumulator
    float acc = 0.0;

    // Stage 1: Serial reduction (reduce to global size)
    for (int i = global_idx; i < n; i += global_len) {
        acc = fmax(acc, fabs(input[i]));
    }

    // Stage 2: Parallel reduction (reduce to #workgroups)
    shared[local_idx] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offs = (local_len >> 1); offs > 0; offs >>= 1) {
        if (local_idx < offs) {
            acc = fmax(acc, shared[local_idx + offs]);
            shared[local_idx] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write-back result
    if (local_idx == 0) {
        output[get_group_id(0)] = shared[0];
    }
}
