//! A port of `numpy.interp` for increasing sample points.
//!
//! The arithmetic follows `arr_interp` in numpy's `compiled_base.c`
//! operation for operation, so results are bit-identical, not just close.

/// Interpolate `fp` over `xp` at `x`, clamping outside `xp`.
///
/// `xp` must be non-empty, increasing and as long as `fp`.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    let left = fp[0];
    let right = fp[n - 1];
    if n == 1 {
        return left;
    }
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return right;
    }
    if x < xp[0] {
        return left;
    }
    // Last j with xp[j] <= x.
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 {
        return fp[j];
    }
    if xp[j] == x {
        // numpy avoids a non-finite slope this way.
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut out = slope * (x - xp[j]) + fp[j];
    if out.is_nan() {
        out = slope * (x - xp[j + 1]) + fp[j + 1];
        if out.is_nan() && fp[j] == fp[j + 1] {
            out = fp[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_and_interpolates() {
        let xp = [0.1, 0.5, 0.9];
        let fp = [1.0, 2.0, 4.0];
        assert_eq!(interp(0.0, &xp, &fp), 1.0);
        assert_eq!(interp(1.0, &xp, &fp), 4.0);
        assert_eq!(interp(0.5, &xp, &fp), 2.0);
        assert_eq!(interp(0.9, &xp, &fp), 4.0);
        assert!((interp(0.7, &xp, &fp) - 3.0).abs() < 1e-12);
    }
}
