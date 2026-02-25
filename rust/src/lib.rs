use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

fn choose_harmonics_grid(n_obs: usize, design_robust: bool, harmonics_grid: &[usize]) -> Vec<usize> {
    if !harmonics_grid.is_empty() {
        return harmonics_grid.to_vec();
    }
    if !design_robust {
        return vec![1, 2, 3];
    }
    if n_obs <= 14 {
        return vec![1, 2];
    }
    if n_obs <= 30 {
        return vec![1, 2, 3];
    }
    vec![1, 2, 3, 4]
}

fn fourier_value(t: f64, period: f64, h: usize, is_sin: bool) -> f64 {
    let w = 2.0 * std::f64::consts::PI * h as f64 / period;
    if is_sin {
        (w * t).sin()
    } else {
        (w * t).cos()
    }
}

fn solve_linear_system(mut a: Vec<f64>, mut b: Vec<f64>, p: usize) -> Option<Vec<f64>> {
    for k in 0..p {
        let mut pivot = k;
        let mut pivot_abs = a[k * p + k].abs();
        for i in (k + 1)..p {
            let v = a[i * p + k].abs();
            if v > pivot_abs {
                pivot = i;
                pivot_abs = v;
            }
        }
        if pivot_abs < 1e-12 {
            return None;
        }
        if pivot != k {
            for j in k..p {
                a.swap(k * p + j, pivot * p + j);
            }
            b.swap(k, pivot);
        }
        let akk = a[k * p + k];
        for i in (k + 1)..p {
            let factor = a[i * p + k] / akk;
            a[i * p + k] = 0.0;
            for j in (k + 1)..p {
                a[i * p + j] -= factor * a[k * p + j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut rhs = b[i];
        for j in (i + 1)..p {
            rhs -= a[i * p + j] * x[j];
        }
        let aii = a[i * p + i];
        if aii.abs() < 1e-12 {
            return None;
        }
        x[i] = rhs / aii;
    }
    Some(x)
}

fn least_squares_yhat(yo: &[f64], to: &[f64], period: f64, harmonics: usize) -> Option<Vec<f64>> {
    let n = yo.len();
    let p = 1 + 2 * harmonics;
    if n == 0 || p == 0 {
        return None;
    }

    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];

    for i in 0..n {
        let ti = to[i];
        let yi = yo[i];
        let mut row = vec![0.0_f64; p];
        row[0] = 1.0;
        for h in 1..=harmonics {
            let j = 2 * h - 1;
            row[j] = fourier_value(ti, period, h, false);
            row[j + 1] = fourier_value(ti, period, h, true);
        }
        for r in 0..p {
            xty[r] += row[r] * yi;
            for c in 0..p {
                xtx[r * p + c] += row[r] * row[c];
            }
        }
    }

    for d in 0..p {
        xtx[d * p + d] += 1e-10;
    }

    let beta = solve_linear_system(xtx, xty, p)?;
    let mut yhat = vec![0.0_f64; n];
    for i in 0..n {
        let ti = to[i];
        let mut v = beta[0];
        for h in 1..=harmonics {
            let j = 2 * h - 1;
            v += beta[j] * fourier_value(ti, period, h, false);
            v += beta[j + 1] * fourier_value(ti, period, h, true);
        }
        yhat[i] = v;
    }
    Some(yhat)
}

fn kendall_tau_b(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    let mut concordant: u64 = 0;
    let mut discordant: u64 = 0;
    let mut ties_x: u64 = 0;
    let mut ties_y: u64 = 0;

    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            if !dx.is_finite() || !dy.is_finite() {
                continue;
            }
            if dx == 0.0 && dy == 0.0 {
                continue;
            } else if dx == 0.0 {
                ties_x += 1;
            } else if dy == 0.0 {
                ties_y += 1;
            } else if dx.signum() == dy.signum() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let s = concordant as f64 - discordant as f64;
    let c_or_d = concordant + discordant;
    let denom_left = c_or_d + ties_x;
    let denom_right = c_or_d + ties_y;
    if denom_left == 0 || denom_right == 0 {
        return None;
    }
    let denom = ((denom_left as f64) * (denom_right as f64)).sqrt();
    if denom <= 0.0 {
        return None;
    }
    Some(s / denom)
}

fn score_tau_series(
    y: &[f64],
    t: &[f64],
    period: f64,
    harmonics_grid: &[usize],
    min_obs: usize,
    design_robust: bool,
) -> Option<f64> {
    if y.len() != t.len() {
        return None;
    }
    let mut yo = Vec::with_capacity(y.len());
    let mut to = Vec::with_capacity(t.len());
    for i in 0..y.len() {
        if y[i].is_finite() && t[i].is_finite() {
            yo.push(y[i]);
            to.push(t[i]);
        }
    }

    if yo.len() < min_obs {
        return None;
    }

    let h_grid = choose_harmonics_grid(yo.len(), design_robust, harmonics_grid);
    let mut best_tau: Option<f64> = None;
    for h in h_grid {
        if h == 0 {
            continue;
        }
        let yhat = match least_squares_yhat(&yo, &to, period, h) {
            Some(v) => v,
            None => continue,
        };
        let tau = match kendall_tau_b(&yo, &yhat) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        if let Some(cur) = best_tau {
            if tau.abs() > cur.abs() {
                best_tau = Some(tau);
            }
        } else {
            best_tau = Some(tau);
        }
    }
    best_tau
}

fn shuffle_in_place(rng: &mut ChaCha8Rng, values: &mut [f64]) {
    let n = values.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        values.swap(i, j);
    }
}

fn seed_for_series(base_seed: u64, series_idx: usize) -> u64 {
    // SplitMix64-style mixing for stable per-series streams.
    let mut z = base_seed ^ (series_idx as u64).wrapping_mul(0x9E3779B97F4A7C15_u64);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9_u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB_u64);
    z ^ (z >> 31)
}

fn compute_perm_scan(
    matrix: &[f64],
    n_rows: usize,
    n_cols: usize,
    t: &[f64],
    period: f64,
    harmonics_grid: &[usize],
    min_obs: usize,
    design_robust: bool,
    n_perm: usize,
    random_seed: u64,
) -> Vec<f64> {
    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = &matrix[(i * n_cols)..((i + 1) * n_cols)];
            let tau_obs = match score_tau_series(row, t, period, harmonics_grid, min_obs, design_robust) {
                Some(v) => v,
                None => return f64::NAN,
            };

            let mut rng = ChaCha8Rng::seed_from_u64(seed_for_series(random_seed, i));
            let mut y_perm = row.to_vec();
            let mut ge_count: usize = 0;
            for _ in 0..n_perm {
                shuffle_in_place(&mut rng, &mut y_perm);
                let tau_b = score_tau_series(&y_perm, t, period, harmonics_grid, min_obs, design_robust)
                    .unwrap_or(0.0)
                    .abs();
                if tau_b >= tau_obs.abs() {
                    ge_count += 1;
                }
            }
            (1.0 + ge_count as f64) / (n_perm as f64 + 1.0)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    matrix,
    t,
    period,
    harmonics_grid,
    min_obs,
    design_robust,
    n_perm,
    random_seed,
    n_threads
))]
fn permutation_pvalues_scan_tau<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    period: f64,
    harmonics_grid: Vec<usize>,
    min_obs: usize,
    design_robust: bool,
    n_perm: usize,
    random_seed: u64,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mat = matrix.as_array();
    let tt = t.as_slice()?;
    let (n_rows, n_cols) = mat.dim();
    if tt.len() != n_cols {
        return Err(PyValueError::new_err(
            "t length must match matrix column count",
        ));
    }
    if n_perm == 0 {
        return Err(PyValueError::new_err("n_perm must be >= 1"));
    }

    let matrix_owned = mat.to_owned().into_raw_vec();
    let compute = || {
        compute_perm_scan(
            &matrix_owned,
            n_rows,
            n_cols,
            tt,
            period,
            &harmonics_grid,
            min_obs,
            design_robust,
            n_perm,
            random_seed,
        )
    };

    let out = if n_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("failed to build rayon pool: {e}")))?;
        pool.install(compute)
    } else {
        compute()
    };

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pymodule]
fn _rust_stats(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permutation_pvalues_scan_tau, m)?)?;
    Ok(())
}
