use global_neutral_euler::physics::*;
fn advance(mut u: Vec<Q>, a: Atmos, dx: f64, dt: f64, steps: usize, muscl: bool) -> Vec<Q> {
    let n = u.len();
    let derivative = |u: &[Q]| {
        let p = u.iter().map(|&q| primitive(q, a)).collect::<Vec<_>>();
        let slope = (0..n)
            .map(|i| {
                if muscl {
                    std::array::from_fn(|f| {
                        mc(p[i][f] - p[(i + n - 1) % n][f], p[(i + 1) % n][f] - p[i][f])
                    })
                } else {
                    [0.; 7]
                }
            })
            .collect::<Vec<Q>>();
        let flux = (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                let left = conserved(std::array::from_fn(|f| p[i][f] + 0.5 * slope[i][f]), a);
                let right = conserved(std::array::from_fn(|f| p[j][f] - 0.5 * slope[j][f]), a);
                hll(left, right, a, a, [1., 0., 0.])
            })
            .collect::<Vec<_>>();
        (0..n)
            .map(|i| std::array::from_fn(|f| -(flux[i][f] - flux[(i + n - 1) % n][f]) / dx))
            .collect::<Vec<Q>>()
    };
    for _ in 0..steps {
        let d = derivative(&u);
        let stage = (0..n)
            .map(|i| std::array::from_fn(|f| u[i][f] + dt * d[i][f]))
            .collect::<Vec<Q>>();
        assert!(stage.iter().all(|&q| admissible(q, a)));
        let d = derivative(&stage);
        for i in 0..n {
            for f in 0..7 {
                u[i][f] = 0.5 * (u[i][f] + stage[i][f] + dt * d[i][f]);
            }
            assert!(admissible(u[i], a));
        }
    }
    u
}
#[test]
fn sod_positive_conservative() {
    let a = Atmos {
        rho: 1.,
        p: 1.,
        r: 1.,
        cv: 1.5,
        g: 0.,
    };
    let u = (0..200)
        .map(|i| {
            conserved(
                if i < 100 {
                    [1., 0., 0., 0., 1., 0., 0.]
                } else {
                    [0.125, 0., 0., 0., 0.1, 0., 0.]
                },
                a,
            )
        })
        .collect::<Vec<_>>();
    let totals = |v: &[Q], f| v.iter().map(|q| q[f]).sum::<f64>();
    let initial = [totals(&u, 0), totals(&u, 4)];
    let result = advance(u, a, 1. / 200., 0.0005, 200, true);
    for (f, start) in [0, 4].into_iter().zip(initial) {
        assert!((totals(&result, f) - start).abs() / start < 1e-12);
    }
}
#[test]
fn muscl_retains_weak_acoustic_wave() {
    let a = Atmos {
        rho: 1.,
        p: 600000.,
        r: 1.,
        cv: 1.5,
        g: 0.,
    };
    let c = 1000.;
    let eps = 1e-6;
    let dx = 2500.;
    let wavelength = 100000.;
    let n = 120;
    let wave = 2. * std::f64::consts::PI / wavelength;
    let u = (0..n)
        .map(|i| {
            let perturb = eps * (wave * i as f64 * dx).cos();
            let rho = 1. + perturb;
            let vel = c * perturb;
            let p = a.p + c * c * perturb;
            [
                rho,
                rho * vel,
                0.,
                0.,
                p * a.cv / a.r + 0.5 * rho * vel * vel,
                0.,
                0.,
            ]
        })
        .collect::<Vec<Q>>();
    let amplitude = |v: &[Q]| {
        let re = (0..n)
            .map(|i| (v[i][0] - 1.) * (wave * i as f64 * dx).cos())
            .sum::<f64>();
        let im = (0..n)
            .map(|i| (v[i][0] - 1.) * (wave * i as f64 * dx).sin())
            .sum::<f64>();
        2. * re.hypot(im) / (n as f64 * eps)
    };
    let low = advance(u.clone(), a, dx, 0.5, 1200, false);
    let high = advance(u, a, dx, 0.5, 1200, true);
    assert!(amplitude(&high) > 0.9);
    assert!(amplitude(&high) > amplitude(&low) * 5.);
}
