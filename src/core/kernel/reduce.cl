//! Dual-Stage Reductions for floating-point values.
//!
//! Based on https://www.cl.cam.ac.uk/teaching/1617/AdvGraph/07_OpenCL.pdf
//!


__kernel void reduce_max_abs(
    __global const float* input,
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
        output[get_group_id(0)] = acc;
    }
}


__kernel void reduce_max(
    __global const float* input,
    __global float* output,
    __local float* shared,
    const uint n
) {
    const int global_idx = get_global_id(0);
    const int global_len = get_global_size(0);
    const int local_idx = get_local_id(0);
    const int local_len = get_local_size(0);

    // Initialize accumulator
    float acc = -INFINITY;

    // Stage 1: Serial reduction (reduce to global size)
    for (int i = global_idx; i < n; i += global_len) {
        acc = fmax(acc, input[i]);
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
        output[get_group_id(0)] = acc;
    }
}


__kernel void reduce_min(
    __global const float* input,
    __global float* output,
    __local float* shared,
    const uint n
) {
    const int global_idx = get_global_id(0);
    const int global_len = get_global_size(0);
    const int local_idx = get_local_id(0);
    const int local_len = get_local_size(0);

    // Initialize accumulator
    float acc = INFINITY;

    // Stage 1: Serial reduction (reduce to global size)
    for (int i = global_idx; i < n; i += global_len) {
        acc = fmin(acc, input[i]);
    }

    // Stage 2: Parallel reduction (reduce to #workgroups)
    shared[local_idx] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offs = (local_len >> 1); offs > 0; offs >>= 1) {
        if (local_idx < offs) {
            acc = fmin(acc, shared[local_idx + offs]);
            shared[local_idx] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write-back result
    if (local_idx == 0) {
        output[get_group_id(0)] = acc;
    }
}


__kernel void reduce_minmax(
    __global const float* input,
    __global float* output,         // size: 2 * #workgroups
    __local float* shared,          // size: 2 * local-size
    const uint n
) {
    const int global_idx = get_global_id(0);
    const int global_len = get_global_size(0);
    const int local_idx = get_local_id(0);
    const int local_len = get_local_size(0);

    // Initialize accumulator
    float acc_min = INFINITY;
    float acc_max = -INFINITY;

    // Stage 1: Serial reduction (reduce to global size)
    for (int i = global_idx; i < n; i += global_len) {
        acc_min = fmin(acc_min, input[i]);
        acc_max = fmax(acc_max, input[i]);
    }

    // Stage 2: Parallel reduction (reduce to #workgroups)
    shared[local_idx] = acc_min;
    shared[local_len + local_idx] = acc_max;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offs = (local_len >> 1); offs > 0; offs >>= 1) {
        if (local_idx < offs) {
            acc_min = fmin(acc_min, shared[local_idx + offs]);
            acc_max = fmax(acc_max, shared[local_len + local_idx + offs]);

            shared[local_idx] = acc_min;
            shared[local_len + local_idx] = acc_max;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write-back result
    if (local_idx == 0) {
        int group_idx = get_group_id(0);
        int group_len = get_num_groups(0);

        output[group_idx] = acc_min;
        output[group_len + group_idx] = acc_max;
    }
}


__kernel void reduce_sum(
    __global const float* input,
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
        acc = acc + input[i];
    }

    // Stage 2: Parallel reduction (reduce to #workgroups)
    shared[local_idx] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offs = (local_len >> 1); offs > 0; offs >>= 1) {
        if (local_idx < offs) {
            acc = acc + shared[local_idx + offs];
            shared[local_idx] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write-back result
    if (local_idx == 0) {
        output[get_group_id(0)] = acc;
    }
}
