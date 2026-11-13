//! Dense-vector helpers shared by the profile roll-up (ingest) and routing.
//! Accumulation is done in f64 so 1024-d sums of many chunk vectors stay accurate.

/// Embedding dimension for every layer (bge-m3 dense head).
pub const DIM: usize = 1024;

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// Scales `v` to unit length in place. No-op on a zero (or non-finite-norm) vector.
pub fn l2_normalize(v: &mut [f32]) {
    let n = norm(v);
    if n == 0.0 || !n.is_finite() {
        return;
    }
    for x in v.iter_mut() {
        *x = (f64::from(*x) / n) as f32;
    }
}

/// Full cosine similarity in `[-1, 1]`; `0.0` if either vector is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine of vectors with different lengths");
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 || !na.is_finite() || !nb.is_finite() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    ((dot / (na * nb)) as f32).clamp(-1.0, 1.0)
}

/// Weighted arithmetic mean (not normalised). `None` if `items` is empty, the total
/// weight is not positive (or NaN), or the vectors have different lengths.
pub fn weighted_mean(items: &[(&[f32], f64)]) -> Option<Vec<f32>> {
    let (first, _) = items.first()?;
    let len = first.len();
    if items.iter().any(|(v, _)| v.len() != len) {
        return None;
    }
    let total: f64 = items.iter().map(|(_, w)| *w).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let mut acc = vec![0.0_f64; len];
    for (v, w) in items {
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += w * f64::from(x);
        }
    }
    Some(acc.into_iter().map(|a| (a / total) as f32).collect())
}

/// Routing profile: `l2norm(α·desc + (1−α)·content)`, or `desc` unchanged when there is
/// no content yet.
pub fn profile(desc: &[f32], content: Option<&[f32]>, alpha: f32) -> Vec<f32> {
    let Some(content) = content else {
        return desc.to_vec();
    };
    debug_assert_eq!(
        desc.len(),
        content.len(),
        "profile of vectors with different lengths"
    );
    let mut v: Vec<f32> = desc
        .iter()
        .zip(content)
        .map(|(&d, &c)| alpha * d + (1.0 - alpha) * c)
        .collect();
    l2_normalize(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn norm_of(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn dim_is_bge_m3() {
        assert_eq!(DIM, 1024);
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!(close(v[0], 0.6) && close(v[1], 0.8), "{v:?}");
        let mut big = vec![1.0_f32; DIM];
        l2_normalize(&mut big);
        assert!(close(norm_of(&big), 1.0));
    }

    #[test]
    fn l2_normalize_is_noop_on_zero_vector() {
        let mut v = vec![0.0_f32; 4];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0; 4]);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn cosine_is_scale_invariant_and_bounded() {
        assert!(close(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
        assert!(close(cosine(&[1.0, 2.0], &[2.0, 4.0]), 1.0));
        assert!(close(cosine(&[1.0, 2.0], &[-3.0, -6.0]), -1.0));
        // Not a dot product: unnormalised inputs still give a cosine.
        assert!(close(
            cosine(&[10.0, 0.0], &[5.0, 5.0]),
            std::f32::consts::FRAC_1_SQRT_2
        ));
        let c = cosine(&[0.1, 0.2, 0.3], &[0.1, 0.2, 0.3]);
        assert!(c <= 1.0 && close(c, 1.0));
    }

    #[test]
    fn cosine_with_zero_vector_is_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[1.0, 1.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn weighted_mean_weights_by_tokens() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let m = weighted_mean(&[(&a[..], 1.0), (&b[..], 3.0)]).unwrap();
        assert!(close(m[0], 0.25) && close(m[1], 0.75), "{m:?}");
    }

    #[test]
    fn weighted_mean_none_cases() {
        assert!(weighted_mean(&[]).is_none());
        let a = [1.0_f32, 0.0];
        assert!(weighted_mean(&[(&a[..], 0.0), (&a[..], 0.0)]).is_none());
        assert!(weighted_mean(&[(&a[..], f64::NAN)]).is_none());
        let short = [1.0_f32];
        assert!(weighted_mean(&[(&a[..], 1.0), (&short[..], 1.0)]).is_none());
    }

    #[test]
    fn profile_without_content_is_desc_clone() {
        let d = vec![2.0_f32, 0.0];
        assert_eq!(profile(&d, None, 0.5), d);
    }

    #[test]
    fn profile_blends_and_normalises() {
        let d = [1.0_f32, 0.0];
        let c = [0.0_f32, 1.0];
        let p = profile(&d, Some(&c[..]), 0.5);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!(close(p[0], h) && close(p[1], h), "{p:?}");
        let p1 = profile(&[2.0, 0.0], Some(&c[..]), 1.0);
        assert!(close(p1[0], 1.0) && close(p1[1], 0.0), "{p1:?}");
        let p0 = profile(&d, Some(&[0.0_f32, 3.0][..]), 0.0);
        assert!(close(p0[0], 0.0) && close(p0[1], 1.0), "{p0:?}");
    }

    #[test]
    fn profile_cancelling_vectors_stay_zero_not_nan() {
        let d = [1.0_f32, 0.0];
        let c = [-1.0_f32, 0.0];
        let p = profile(&d, Some(&c[..]), 0.5);
        assert_eq!(p, vec![0.0, 0.0]);
    }
}
