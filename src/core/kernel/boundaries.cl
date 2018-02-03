//! Kernels for boundary conditions.
//!
//!
//! Boundary bit format (8 bit):
//!
//!   Bits: |x|x|x|x|xxxx|
//!   Name: |l|r|b|t|self|
//!
//!   with
//!
//!   l: 1 bit, indicates fluid in left cell (1) or solid (0).
//!   r: 1 bit, indicates fluid in right cell (1) or solid (0).
//!   b: 1 bit, indicates fluid in cell below (1) or solid (0).
//!   t: 1 bit, indicates fluid in cell above (1) or solid (0).
//!   self: 4 bit, see below.
//!
//!
//! Self-tag:
//!
//!   Bits: |x|x|xx|
//!   Name: |v|h|ty|
//!
//!   with
//!
//!   v: 1 bit, boundary applies vertical (when 1).
//!   h: 1 bit, boundary applies horizontal (when 1).
//!   ty: 2 bit, boundary type (see below).
//!
//!
//! Boundary type:
//!
//!   Name          |ty|  Notes
//!   -----------------------------------
//!   Fluid/NoSlip  |00|  depends on v, h
//!   Inflow        |01|
//!   Slip/Outflow  |10|  depends on v, h
//!
//!   Note: The type restricts the valid values for v and h.
//!
//!
//! Valid self-tag combinations:
//!
//!   Name          |v|h|ty|
//!   ----------------------
//!   Fluid         |0|0|00|
//!   NoSlip        |1|1|00|
//!   Inflow        |1|1|01|
//!   InflowHoriz   |0|1|01|
//!   InflowVert    |1|0|01|
//!   SlipHoriz     |0|1|10|
//!   SlipVert      |1|0|10|
//!   Outflow       |1|1|10|
//!
//!   Other combinations are not allowed.
//!
//!
//! Important: Outer boundary cells must not have neighbor-is-fluid bits set
//! for their respective boundary!
//!


#define BC_MASK_NEIGHBOR_LEFT           0b10000000
#define BC_MASK_NEIGHBOR_RIGHT          0b01000000
#define BC_MASK_NEIGHBOR_BOTTOM         0b00100000
#define BC_MASK_NEIGHBOR_TOP            0b00010000

#define BC_MASK_SELF                    0b00001111
#define BC_MASK_SELF_ORIENTATION        0b00001100
#define BC_MASK_SELF_VERT               0b00001000
#define BC_MASK_SELF_HORZ               0b00000100
#define BC_MASK_SELF_TYPE               0b00000011

#define BC_TYPE_NOSLIP                  0b00000000      // Fluid, NoSlip
#define BC_TYPE_VELOCITY                0b00000001      // Inflow, InflowHoriz, InflowVert
#define BC_TYPE_PRESSURE                0b00000010      // SlipHoriz, SlipVert, Outflow


#define BC_IS_NEIGHBOR_LEFT_FLUID(x)    ((x) & (BC_MASK_NEIGHBOR_LEFT))
#define BC_IS_NEIGHBOR_RIGHT_FLUID(x)   ((x) & (BC_MASK_NEIGHBOR_RIGHT))
#define BC_IS_NEIGHBOR_BOTTOM_FLUID(x)  ((x) & (BC_MASK_NEIGHBOR_BOTTOM))
#define BC_IS_NEIGHBOR_TOP_FLUID(x)     ((x) & (BC_MASK_NEIGHBOR_TOP))

#define BC_IS_SELF_VERT(x)              ((x) & (BC_MASK_SELF_VERT))
#define BC_IS_SELF_HORZ(x)              ((x) & (BC_MASK_SELF_HORZ))

#define BC_SELF_FLUID                   0b0000
#define BC_SELF_NOSLIP                  0b1100
#define BC_SELF_INFLOW                  0b1101
#define BC_SELF_INFLOW_H                0b0101
#define BC_SELF_INFLOW_V                0b1001
#define BC_SELF_SLIP_H                  0b0110
#define BC_SELF_SLIP_V                  0b1010
#define BC_SELF_OUTFLOW                 0b1110


#define BC_MASKED_EQUALS(x, mask)       (((mask) & (x)) == (mask))


//! Converts a two-dimensional index to a linear index.
#define INDEX(x, y, size_x) (((y) * (size_x)) + (x))


__kernel void set_boundary_u(
    __read_write __global float* u,     // (n + 3) * (m + 2)
    __read_only __global uchar b,       // (n + 2) * (m + 2)
    __read_only float u_in
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n + 2, len.y = m + 2
    // NOTE: assumes pos is valid for b

    const int b_size_x = get_global_size(0);        // equals n + 2 in doc above
    const int u_size_x = b_size_x + 1;

    // load boundary type
    const uchar b_cell = b[INDEX(pos.x, pos.y, size_x)];
    const uchar b_self = b_cell & BC_MASK_SELF;

    // get orientation of boundary
    const int2 u_pos = (int2)(pos.x + 1, pos.y);    // handle additional cells for domain subdivision
    int2 u_pos_inner;
    bool on_boundary;

    // u on left boundary
    if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_RIGHT_FLUID(b_cell)) {
        u_pos_inner = (int2)(u_pos.x + 1, u_pos.y);
        on_boundary = true;

    // u on right boundary
    } else if (b_self == BC_SELF_FLUID && !BC_IS_NEIGHBOR_RIGHT_FLUID(b_cell)) {
        u_pos_inner = (int2)(u_pos.x - 1, u_pos.y);
        on_boundary = true;

    // u below bottom boundary
    } else if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_TOP_FLUID(b_cell)) {
        u_pos_inner = (int2)(u_pos.x, u_pos.y + 1);
        on_boundary = false;

    // u above top boundary
    } else if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_BELOW_FLUID(b_cell)) {
        u_pos_inner = (int2)(u_pos.x, u_pos.y - 1);
        on_boundary = false;

    // u is solid cell fully enclosed by other solid cells
    } else if (b_self != BC_SELF_FLUID) {
        u[INDEX(u_pos.x, u_pos.y, u_size_x)] = 0.0;
        return;

    // u is fluid cell fully enclosed by other fluid cells
    } else {
        return;
    }

    // velocity-based inflow/outflow in horizontal direction
    if (BC_MASKED_EQUALS(b_self, BC_TYPE_VELOCITY | BC_MASK_SELF_HORZ)) {
        if (on_boundary) {
            u[INDEX(u_pos.x, u_pos.y, u_size_x)] = u_in;
        } else {
            float u_inner = u[INDEX(u_pos_inner.x, u_pos_inner.y, u_size_x)];
            u[INDEX(u_pos.x, u_pos.y, u_size_x)] = 2.0 * u_in - u_inner;
        }

    // pressure-based inflow/outflow in horizontal direction
    } else if (BC_MASKED_EQUALS(b_self, BC_TYPE_PRESSURE | BC_MASK_SELF_HORZ)) {
        float u_inner = u[INDEX(u_pos_inner.x, u_pos_inner.y, u_size_x)];
        u[INDEX(u_pos.x, u_pos.y, u_size_x)] = u_inner;

    // boundary is no-slip w.r.t. horizontal flow
    } else {
        if (on_boundary) {
            u[INDEX(u_pos.x, u_pos.y, u_size_x)] = 0.0;
        } else {
            float u_inner = u[INDEX(u_pos_inner.x, u_pos_inner.y, u_size_x)];
            u[INDEX(u_pos.x, u_pos.y, u_size_x)] = -u_inner;
        }
    }
}


__kernel void set_boundary_v(
    __read_write __global float* v,     // (n + 2) + (m + 3)
    __read_only __global uchar b,       // (n + 2) * (m + 2)
    __read_only float v_in
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n + 2, len.y = m + 2
    // NOTE: pos is valid for b

    const int b_size_x = get_global_size(0);        // equals n + 2 in doc above
    const int v_size_x = b_size_x;

    // load boundary type
    const uchar b_cell = b[INDEX(pos.x, pos.y, b_size_x)];
    const uchar b_self = b_cell & BC_MASK_SELF;

    // get orientation of boundary
    const int2 v_pos = (int2)(pos.x, pos.y + 1);    // handle additional cells for domain subdivision
    int2 v_pos_inner;
    bool on_boundary;

    // v on bottom boundary
    if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_TOP_FLUID(b_cell)) {
        v_pos_inner = (int2)(v_pos.x, v_pos.y + 1);
        on_boundary = true;

    // v on top boundary
    } else if (b_self == BC_SELF_FLUID && !BC_IS_NEIGHBOR_TOP_FLUID(b_cell)) {
        v_pos_inner = (int2)(v_pos.x, v_pos.y - 1);
        on_boundary = true;

    // v left of left boundary
    } else if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_RIGHT_FLUID(b_cell)) {
        v_pos_inner = (int2)(v_pos.x + 1, v_pos.y);
        on_boundary = false;

    // v right of right boundary
    } else if (b_self != BC_SELF_FLUID && BC_IS_NEIGHBOR_LEFT_FLUID(b_cell)) {
        v_pos_inner = (int2)(v_pos.x - 1, v_pos.y);
        on_boundary = false;

    // v is solid cell fully enclosed by other solid cells
    } else if (b_self != BC_SELF_FLUID) {
        u[INDEX(v_pos.x, v_pos.y, b_size_x)] = 0.0;
        return;

    // v is fluid cell fully enclosed by other fluid cells
    } else {
        return;
    }

    // velocity-based inflow/outflow in vertical direction
    if (BC_MASKED_EQUALS(b_self, BC_TYPE_VELOCITY | BC_MASK_SELF_VERT)) {
        if (on_boundary) {
            v[INDEX(v_pos.x, v_pos.y, v_size_x)] = v_in;
        } else {
            float v_inner = v[INDEX(v_pos_inner.x, v_pos_inner.y, v_size_x)];
            v[INDEX(v_pos.x, v_pos.y, v_size_x)] = 2.0 * v_in - v_inner;
        }

    // pressure-based inflow/outflow in vertical direction
    } else if (BC_MASKED_EQUALS(b_self, BC_TYPE_PRESSURE | BC_MASK_SELF_VERT)) {
        float v_inner = v[INDEX(v_pos_inner.x, v_pos_inner.y, v_size_x)];
        v[INDEX(v_pos.x, v_pos.y, v_size_x)] = v_inner;

    // boundary is no-slip w.r.t. vertical flow
    } else {
        if (on_boundary) {
            v[INDEX(v_pos.x, v_pos.y, v_size_x)] = 0.0;
        } else {
            float v_inner = v[INDEX(v_pos_inner.x, v_pos_inner.y, v_size_x)];
            v[INDEX(v_pos.x, v_pos.y, v_size_x)] = -v_inner;
        }
    }
}


__kernel void set_boundary_p(
    __read_write __global float* p,     // (n + 2) * (m + 2)
    __read_only __global uchar b,       // (n + 2) * (m + 2)
    __read_only float p_in,
) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // assumes: pos.x >= 0 && pos.x <= (len - 1) && pos.y >= 0 && pos.y <= (len - 1)
    // with len.x = n + 2, len.y = m + 2

    const int size_x = get_global_size(0);    // equals n + 2 in doc above

    // load boundary type
    const uchar b_cell = b[INDEX(pos.x, pos.y, size_x)];
    const uchar b_self = b_cell & BC_MASK_SELF;

    // don't do anything if this is a fluid cell
    if (b_self == BC_SELF_FLUID) {
        return;
    }

    float n = 0.0;
    float v = 0.0;

    // left of left boundary
    if (BC_IS_NEIGHBOR_RIGHT_FLUID(b_cell)) {
        float p_inner = p[INDEX(pos.x + 1, pos.y, size_x)];
        n += 1.0;

        if (b_self == BC_SELF_OUTFLOW) {
            v += /* 2.0 * 0.0 */ - p_inner;                 // fixed pressure boundary (p = 0)
        } else if (b_self == BC_SELF_SLIP_H) {
            v += 2.0 * p_in - p_inner;                      // fixed pressure boundary
        } else {
            v += p_inner;                                   // solid boundary
        }
    }

    // right of right boundary
    if (BC_IS_NEIGHBOR_LEFT_FLUID(b_cell)) {
        float p_inner = p[INDEX(pos.x - 1, pos.y, size_x)];
        n += 1.0;

        if (b_self == BC_SELF_OUTFLOW) {
            v += /* 2.0 * 0.0 */ - p_inner;                 // fixed pressure boundary (p = 0)
        } else if (b_self == BC_SELF_SLIP_H) {
            v += 2.0 * p_in - p_inner;                      // fixed pressure boundary
        } else {
            v += p_inner;                                   // solid boundary
        }
    }

    // below bottom boundary
    if (BC_IS_NEIGHBOR_TOP_FLUID(b_cell)) {
        float p_inner = p[INDEX(pos.x, pos.y + 1, size_x)];
        n += 1.0;

        if (b_self == BC_SELF_OUTFLOW) {
            v += /* 2.0 * 0.0 */ - p_inner;                 // fixed pressure boundary (p = 0)
        } else if (b_self == BC_SELF_SLIP_V) {
            v += 2.0 * p_in - p_inner;                      // fixed prssure boundary
        } else {
            v += p_inner;                                   // solid boundary
        }
    }

    // above top boundary
    if (BC_IS_NEIGHBOR_BOTTOM_FLUID(b_cell)) {
        float p_inner = p[INDEX(pos.x, pos.y - 1, size_x)];
        n += 1.0;

        if (b_self == BC_SELF_OUTFLOW) {
            v += /* 2.0 * 0.0 */ - p_inner;                 // fixed pressure boundary (p = 0)
        } else if (b_self == BC_SELF_SLIP_V) {
            v += 2.0 * p_in - p_inner;                      // fixed pressure boundary
        } else {
            v += p_inner;                                   // solid boundary
        }
    }

    // store result
    if (n > 0.0) {
        p[INDEX(pos.x, pos.y, size_x)] = v / n;
    }
}
