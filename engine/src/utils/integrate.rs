/// Simple trapezoidal integration over paired samples.
pub fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert!(x.len() == y.len() && x.len() >= 2, "inputs must align");
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xw, yw)| {
            let h = xw[1] - xw[0];
            0.5 * h * (yw[0] + yw[1])
        })
        .sum()
}

/// Integrate with an additional weight per sample (e.g., area element).
pub fn trapezoid_with_weights(x: &[f64], f: &[f64], w: &[f64]) -> f64 {
    assert!(
        x.len() == f.len() && f.len() == w.len(),
        "inputs must align"
    );
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let h = x[i + 1] - x[i];
        let f0 = f[i] * w[i];
        let f1 = f[i + 1] * w[i + 1];
        acc += 0.5 * h * (f0 + f1);
    }
    acc
}

/// Integrate a pre-weighted function along a curve by averaging midpoints.
pub fn integrate_weighted(x: &[f64], weighted: &[f64]) -> f64 {
    assert!(x.len() == weighted.len(), "inputs must align");
    x.windows(2)
        .zip(weighted.windows(2))
        .map(|(xw, fw)| {
            let h = xw[1] - xw[0];
            0.5 * h * (fw[0] + fw[1])
        })
        .sum()
}
