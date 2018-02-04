//! Kernels for zero-initialization.
//!


//! Initialize the given float-buffer with zeros.
__kernel void zero_float(__global float* buf) {
    buf[get_global_id(0)] = 0.0;
}
