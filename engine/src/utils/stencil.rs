/// First-order central difference for uniformly spaced samples.
pub fn central_difference(samples: &[f64], step: f64) -> Vec<f64> {
    let n = samples.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mut out = vec![0.0; n];
    for i in 1..n - 1 {
        out[i] = (samples[i + 1] - samples[i - 1]) / (2.0 * step);
    }
    out[0] = (samples[1] - samples[0]) / step;
    out[n - 1] = (samples[n - 1] - samples[n - 2]) / step;
    out
}

/// First derivative for samples on a non-uniform grid using forward/backward edges.
pub fn first_derivative(x: &[f64], f: &[f64]) -> Vec<f64> {
    assert!(x.len() == f.len(), "inputs must align");
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut df = vec![0.0; n];
    for i in 1..n - 1 {
        let h_prev = x[i] - x[i - 1];
        let h_next = x[i + 1] - x[i];
        df[i] = (h_next * (f[i] - f[i - 1]) / h_prev + h_prev * (f[i + 1] - f[i]) / h_next)
            / (h_prev + h_next);
    }
    df[0] = (f[1] - f[0]) / (x[1] - x[0]);
    df[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
    df
}

/// Second derivative via central finite differences on a non-uniform grid.
pub fn second_derivative(x: &[f64], f: &[f64]) -> Vec<f64> {
    assert!(x.len() == f.len(), "inputs must align");
    let n = x.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mut d2f = vec![0.0; n];
    for i in 1..n - 1 {
        let h_prev = x[i] - x[i - 1];
        let h_next = x[i + 1] - x[i];
        let denom = 0.5 * (h_prev + h_next) * h_prev * h_next;
        let term_prev = (f[i] - f[i - 1]) * h_next;
        let term_next = (f[i + 1] - f[i]) * h_prev;
        d2f[i] = 2.0 * (term_next - term_prev) / (denom.max(std::f64::EPSILON));
    }
    d2f[0] = d2f[1];
    d2f[n - 1] = d2f[n - 2];
    d2f
}
