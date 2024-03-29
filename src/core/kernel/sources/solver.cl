//! Red-black SOR solver.
//!


//! Cell color definitions:
//!   (pos.x & 1) == (pos.y & 1)  => red
//!   (pos.x & 1) != (pos.y & 1)  => black
//!   thus: (0,0) => red


#define BC_MASK_SELF                    0b00001111
#define BC_SELF_FLUID                   0b0000


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


__kernel void cycle_red(
    __global float* p,              // (n + 2) * (m + 2)
    __global const float* rhs,      // n * m
    __global const uchar* b,        // (n + 2) * (m + 2)
    const float2 h,
    const float omega
) {
    const int2 rb_pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: rb_pos.x >= 0 && rb_pos.x <= (len - 1) && rb_pos.y >= 0 && rb_pos.y <= (len - 1)
    // with len.x = n, len.y = red(m)

    const int rb_pos_offs_y = rb_pos.x & 1;
    const int2 pos = (int2)(rb_pos.x, rb_pos.y * 2 + rb_pos_offs_y);

    const int rhs_size_x = get_global_size(0);
    const int p_size_x = rhs_size_x + 2;
    const int b_size_x = p_size_x;

    // make sure that this is a fluid cell
    const uchar b_self = b[INDEX(pos.x + 1, pos.y + 1, b_size_x)] & BC_MASK_SELF;
    if (b_self != BC_SELF_FLUID) {
        return;
    }

    // load p
    const float p_cell   = p[INDEX(pos.x + 1, pos.y + 1, p_size_x)];
    const float p_left   = p[INDEX(pos.x    , pos.y + 1, p_size_x)];
    const float p_right  = p[INDEX(pos.x + 2, pos.y + 1, p_size_x)];
    const float p_down   = p[INDEX(pos.x + 1, pos.y    , p_size_x)];
    const float p_top    = p[INDEX(pos.x + 1, pos.y + 2, p_size_x)];

    // load rhs
    const float rhs_cell = rhs[INDEX(pos.x, pos.y, rhs_size_x)];

    // calculate
    const float hx2 = h.x * h.x;
    const float hy2 = h.y * h.y;

    const float dpx = (p_left + p_right) / hx2;
    const float dpy = (p_down + p_top) / hy2;

    const float corr = ((hx2 * hy2) / (2.0 * (hx2 + hy2))) * (dpx + dpy - rhs_cell);

    const float val = (1.0 - omega) * p_cell + omega * corr;
    p[INDEX(pos.x + 1, pos.y + 1, p_size_x)] = val;
}


__kernel void cycle_black(
    __global float* p,              // (n + 2) * (m + 2)
    __global const float* rhs,      // n * m
    __global const uchar* b,        // (n + 2) * (m + 2)
    const float2 h,
    const float omega
) {
    const int2 rb_pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: rb_pos.x >= 0 && rb_pos.x <= (len - 1) && rb_pos.y >= 0 && rb_pos.y <= (len - 1)
    // with len.x = n, len.y = black(m)

    const int rb_pos_offs_y = 1 - (rb_pos.x & 1);
    const int2 pos = (int2)(rb_pos.x, rb_pos.y * 2 + rb_pos_offs_y);

    const int rhs_size_x = get_global_size(0);
    const int p_size_x = rhs_size_x + 2;
    const int b_size_x = p_size_x;

    // make sure that this is a fluid cell
    const uchar b_self = b[INDEX(pos.x + 1, pos.y + 1, b_size_x)] & BC_MASK_SELF;
    if (b_self != BC_SELF_FLUID) {
        return;
    }

    // load p
    const float p_cell   = p[INDEX(pos.x + 1, pos.y + 1, p_size_x)];
    const float p_left   = p[INDEX(pos.x    , pos.y + 1, p_size_x)];
    const float p_right  = p[INDEX(pos.x + 2, pos.y + 1, p_size_x)];
    const float p_down   = p[INDEX(pos.x + 1, pos.y    , p_size_x)];
    const float p_top    = p[INDEX(pos.x + 1, pos.y + 2, p_size_x)];

    // load rhs
    const float rhs_cell = rhs[INDEX(pos.x, pos.y, rhs_size_x)];

    // calculate
    const float hx2 = h.x * h.x;
    const float hy2 = h.y * h.y;

    const float dpx = (p_left + p_right) / hx2;
    const float dpy = (p_down + p_top) / hy2;

    const float corr = ((hx2 * hy2) / (2.0 * (hx2 + hy2))) * (dpx + dpy - rhs_cell);

    const float val = (1.0 - omega) * p_cell + omega * corr;
    p[INDEX(pos.x + 1, pos.y + 1, p_size_x)] = val;
}


__kernel void residual(
    __global const float* p,        // (n + 2) * (m + 2)
    __global const float* rhs,      // n * m
    __global const uchar* b,        // (n + 2) * (m + 2)
    __global float* res,            // n * m
    const float2 h
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n, len.y = m

    const int res_size_x = get_global_size(0);
    const int rhs_size_x = res_size_x;
    const int p_size_x = rhs_size_x + 2;
    const int b_size_x = p_size_x;

    // check if this is a fluid cell
    const uchar b_self = b[INDEX(pos.x + 1, pos.y + 1, b_size_x)] & BC_MASK_SELF;
    if (b_self != BC_SELF_FLUID) {
        res[INDEX(pos.x, pos.y, res_size_x)] = 0.0;
        return;
    }

    // load p
    const float p_cell   = p[INDEX(pos.x + 1, pos.y + 1, p_size_x)];
    const float p_left   = p[INDEX(pos.x    , pos.y + 1, p_size_x)];
    const float p_right  = p[INDEX(pos.x + 2, pos.y + 1, p_size_x)];
    const float p_down   = p[INDEX(pos.x + 1, pos.y    , p_size_x)];
    const float p_top    = p[INDEX(pos.x + 1, pos.y + 2, p_size_x)];

    // load rhs
    const float rhs_cell = rhs[INDEX(pos.x, pos.y, rhs_size_x)];

    // calculate residual
    const float p_dxx = (p_right - 2.0 * p_cell + p_left) / (h.x * h.x);
    const float p_dyy = (p_top - 2.0 * p_cell + p_down) / (h.y * h.y);

    const float val = p_dxx + p_dyy - rhs_cell;

    // store result
    res[INDEX(pos.x, pos.y, res_size_x)] = val * val;
}
