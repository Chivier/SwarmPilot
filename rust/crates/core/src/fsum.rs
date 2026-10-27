//! A port of Python's `math.fsum` (Shewchuk's exact summation).

/// Sum `values` with a single rounding, as `math.fsum` does.
///
/// Returns `None` where Python raises: `OverflowError` when an
/// intermediate sum of finite values overflows, `ValueError` for
/// `-inf + inf`. Other non-finite inputs are summed as Python does.
pub fn fsum(values: &[f64]) -> Option<f64> {
    let mut p: Vec<f64> = Vec::with_capacity(values.len());
    let mut special = 0.0_f64;
    let mut inf_sum = 0.0_f64;
    for &v in values {
        let mut x = v;
        let mut i = 0;
        for j in 0..p.len() {
            let mut y = p[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                p[i] = lo;
                i += 1;
            }
            x = hi;
        }
        p.truncate(i);
        if x != 0.0 {
            if !x.is_finite() {
                if v.is_finite() {
                    return None;
                }
                if v.is_infinite() {
                    inf_sum += v;
                }
                special += v;
                p.clear();
            } else {
                p.push(x);
            }
        }
    }
    if special != 0.0 {
        if inf_sum.is_nan() {
            return None;
        }
        return Some(special);
    }
    let mut n = p.len();
    let mut hi = 0.0;
    if n > 0 {
        n -= 1;
        hi = p[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = p[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Some(hi)
}

#[cfg(test)]
mod tests {
    use super::fsum;

    #[test]
    fn exact_where_naive_sum_is_not() {
        assert_eq!(fsum(&[0.1; 10]), Some(1.0));
        assert_eq!(fsum(&[1e100, 1.0, -1e100, 1e-100]), Some(1.0));
        assert_eq!(fsum(&[]), Some(0.0));
        assert_eq!(fsum(&[f64::MAX, f64::MAX]), None);
        assert_eq!(fsum(&[f64::INFINITY, f64::NEG_INFINITY]), None);
    }
}
