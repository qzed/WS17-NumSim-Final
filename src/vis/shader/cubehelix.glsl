//! Cubehelix colormap as described by D. A. Green [0].
//! To be included in other shaders.
//!
//! [0] : A colour scheme for the display of astronomical intensity images
//!       D. A. Green
//!       Bull. Astr. Soc. India (2011) 39, 289–295
//!
//!
//! Parameters:
//! - start:     the starting color.
//! - rotations: the number of rotations in color.
//! - hue:       controls the saturation.
//! - gamma:     can be used to emphasize low (gamma < 1) or high (gamma > 1) intensity values.
//! - value:     the value to be mapped.
//!
//! Known-good values for the colormap (as [start, rotations, hue]):
//! - Official default:     [+0.5, -1.5, +1.0]
//! - Blue to green:        [+0.3, -0.5, +0.9]
//!

vec3 cubehelix(float start, float rotations, float hue, float gamma, float value) {
    float value_emph = pow(value, gamma);

    float phi = 6.283185307179586 * (start / 3 + rotations * value);
    float amp = hue * value_emph * (1.0 - value_emph) / 2.0;

    mat2x3 coef = mat2x3(
        vec3(-0.14861, -0.29227, +1.97294),
        vec3(+1.78277, -0.90649, +0.00000)
    );

    return value_emph + amp * coef * vec2(cos(phi), sin(phi));
}
