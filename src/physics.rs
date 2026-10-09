use crate::mesh::*;
pub type Q = [f64; 7]; // density, ECEF momentum (3), total energy, CH4/O2 densities
#[derive(Clone, Copy)]
pub struct Atmos {
    pub rho: f64,
    pub p: f64,
    pub r: f64,
    pub cv: f64,
    pub g: f64,
}
const KB: f64 = 1.380649e-23;
const AMU: f64 = 1.66053906660e-27;
const RCH: f64 = KB / (16.043 * AMU);
const RO2: f64 = KB / (31.998 * AMU);
pub fn base(a: Atmos) -> Q {
    [a.rho, 0., 0., 0., a.p * a.cv / a.r, 0., 0.]
}
pub fn thermo(q: Q, a: Atmos) -> (f64, f64) {
    let ambient = q[0] - q[5] - q[6];
    let kin = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) / (2. * q[0]);
    let cv = ambient * a.cv + q[5] * 3. * RCH + q[6] * 2.5 * RO2;
    let gas = ambient * a.r + q[5] * RCH + q[6] * RO2;
    let p = (q[4] - kin) * gas / cv;
    (p, ((1. + gas / cv) * p / q[0]).sqrt())
}
pub fn primitive(q: Q, a: Atmos) -> Q {
    [
        q[0] / a.rho,
        q[1] / q[0],
        q[2] / q[0],
        q[3] / q[0],
        thermo(q, a).0 / a.p,
        q[5] / q[0],
        q[6] / q[0],
    ]
}
pub fn conserved(p: Q, a: Atmos) -> Q {
    let rho = p[0] * a.rho;
    let mut q = [
        rho,
        rho * p[1],
        rho * p[2],
        rho * p[3],
        0.,
        rho * p[5],
        rho * p[6],
    ];
    let ambient = rho - q[5] - q[6];
    let cv = ambient * a.cv + q[5] * 3. * RCH + q[6] * 2.5 * RO2;
    let gas = ambient * a.r + q[5] * RCH + q[6] * RO2;
    q[4] = p[4] * a.p * cv / gas + 0.5 * rho * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
    q
}
pub fn admissible(q: Q, a: Atmos) -> bool {
    q.iter().all(|x| x.is_finite())
        && q[0] > 0.
        && q[5] >= 0.
        && q[6] >= 0.
        && q[5] + q[6] < q[0]
        && thermo(q, a).0 > 0.
}
pub fn flux(q: Q, a: Atmos, n: V) -> Q {
    let p = thermo(q, a).0;
    let vel = dot([q[1], q[2], q[3]], n) / q[0];
    let mut f = q.map(|x| x * vel);
    for d in 0..3 {
        f[d + 1] += (p - a.p) * n[d];
    }
    f[4] = (q[4] + p) * vel;
    f
}
// Same stratified face mapping and HLLE wave-speed estimate as frozen C++.
pub fn hll(left: Q, right: Q, al: Atmos, ar: Atmos, n: V) -> Q {
    hll_at(left, right, al, ar, n, 0.5)
}
/// Face position as a fraction of the physical distance between cell centres.
pub fn hll_at(left: Q, right: Q, al: Atmos, ar: Atmos, n: V, fraction: f64) -> Q {
    // Preserve the supplied resting background exactly rather than creating
    // roundoff-sized fluxes through repeated EOS reconstruction.
    if left == base(al) && right == base(ar) {
        return [0.; 7];
    }
    let mut af = Atmos {
        rho: (al.rho * ar.rho).sqrt(),
        p: (al.p * ar.p).sqrt(),
        r: (al.r + ar.r) / 2.,
        cv: (al.cv + ar.cv) / 2.,
        g: (al.g + ar.g) / 2.,
    };
    // Keep the existing uniform-grid arithmetic exactly at the midpoint.
    if fraction != 0.5 {
        let interp = |l: f64, r: f64| l + fraction * (r - l);
        af = Atmos {
            rho: interp(al.rho.ln(), ar.rho.ln()).exp(),
            p: interp(al.p.ln(), ar.p.ln()).exp(),
            r: interp(al.r, ar.r),
            cv: interp(al.cv, ar.cv),
            g: interp(al.g, ar.g),
        };
    }
    let l = if al.rho == ar.rho && al.p == ar.p && al.r == ar.r && al.cv == ar.cv {
        left
    } else {
        conserved(primitive(left, al), af)
    };
    let r = if al.rho == ar.rho && al.p == ar.p && al.r == ar.r && al.cv == ar.cv {
        right
    } else {
        conserved(primitive(right, ar), af)
    };
    let (pl, cl) = thermo(l, af);
    let (pr, cr) = thermo(r, af);
    assert!(pl > 0. && pr > 0.);
    let ul = dot([l[1], l[2], l[3]], n) / l[0];
    let ur = dot([r[1], r[2], r[3]], n) / r[0];
    let sl = 0f64.min(ul - cl).min(ur - cr);
    let sr = 0f64.max(ul + cl).max(ur + cr);
    let fl = flux(l, af, n);
    let fr = flux(r, af, n);
    std::array::from_fn(|f| (sr * fl[f] - sl * fr[f] + sl * sr * (r[f] - l[f])) / (sr - sl))
}
pub fn mc(l: f64, r: f64) -> f64 {
    if l * r <= 0. {
        0.
    } else {
        l.signum() * (2. * l.abs()).min(2. * r.abs()).min(((l + r) / 2.).abs())
    }
}
pub fn material_rate(rate: f64, k: f64, velocity: V, internal: f64) -> Q {
    [
        rate * k,
        rate * k * velocity[0],
        rate * k * velocity[1],
        rate * k * velocity[2],
        rate * k * (0.5 * dot(velocity, velocity) + internal),
        rate * k / 4.6,
        rate * k * 3.6 / 4.6,
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixture_roundtrip() {
        let a = Atmos {
            rho: 1.,
            p: 86100.,
            r: 287.,
            cv: 717.5,
            g: 9.81,
        };
        for p in [
            [1., 0., 0., 0., 1., 0., 0.],
            [1.2, 7600., 2., 4., 1.3, 0.02, 0.08],
        ] {
            let q = conserved(p, a);
            assert!(admissible(q, a));
            for (x, y) in primitive(q, a).iter().zip(p) {
                assert!((x - y).abs() < 1e-10 * y.abs().max(1.));
            }
        }
    }
    #[test]
    fn flux_reversal() {
        let a = Atmos {
            rho: 1.,
            p: 1.,
            r: 1.,
            cv: 1.5,
            g: 0.,
        };
        let l = base(a);
        let r = conserved([0.125, 0., 0., 0., 0.1, 0., 0.], a);
        let n = unit([1., 2., 3.]);
        let f = hll(l, r, a, a, n);
        let b = hll(r, l, a, a, mul(n, -1.));
        for i in 0..7 {
            assert!((f[i] + b[i]).abs() < 1e-14);
        }
    }
    #[test]
    fn injection_budget() {
        let s = material_rate(1., 1., [7600., 0., 0.], 0.);
        assert_eq!(s[4], 28_880_000.);
        assert!((s[5] + s[6] - s[0]).abs() < 1e-15);
    }
}
