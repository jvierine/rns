//! Implicit Newtonian deviatoric stress and Fourier conduction on actual prisms.
//! A least-squares Cartesian gradient and its exact volume adjoint form an SPD
//! operator. This is an unstructured discretization, not a copied lat/lon stencil.
use crate::{
    mesh::*,
    parallel::{Domain, ExchangeWorkspace},
    physics::*,
    solver::Model,
};
use anyhow::{Result, ensure};
use mpi::{collective::SystemOperation, traits::*};
use std::collections::BTreeSet;
#[derive(Clone)]
struct Quadrature {
    index: usize,
    volume: f64,
    pairs: Vec<(usize, V)>,
    layer: usize,
}
fn inverse(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let rows = a.map(|r| r);
    let det = dot(rows[0], cross(rows[1], rows[2]));
    if det.abs() < 1e-18 {
        // At a closed regional corner there may be only one active horizontal
        // neighbour. Moore-Penrose inverse sets the unsupported derivative to
        // zero, without creating fictitious exterior stresses or heat flux.
        let mut a = a;
        let mut v = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        for _ in 0..40 {
            let (p, q) = [(0, 1), (0, 2), (1, 2)]
                .into_iter()
                .max_by(|&(i, j), &(k, l)| a[i][j].abs().total_cmp(&a[k][l].abs()))
                .unwrap();
            if a[p][q].abs() < 1e-15 {
                break;
            }
            let theta = 0.5 * (2. * a[p][q]).atan2(a[q][q] - a[p][p]);
            let c = theta.cos();
            let s = theta.sin();
            for k in 0..3 {
                let x = a[k][p];
                let y = a[k][q];
                a[k][p] = c * x - s * y;
                a[k][q] = s * x + c * y;
            }
            for k in 0..3 {
                let x = a[p][k];
                let y = a[q][k];
                a[p][k] = c * x - s * y;
                a[q][k] = s * x + c * y;
            }
            for k in 0..3 {
                let x = v[k][p];
                let y = v[k][q];
                v[k][p] = c * x - s * y;
                v[k][q] = s * x + c * y;
            }
        }
        let scale = (0..3).map(|i| a[i][i].abs()).fold(0f64, f64::max);
        return std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..3)
                    .filter(|&k| a[k][k] > scale * 1e-12)
                    .map(|k| v[i][k] * v[j][k] / a[k][k])
                    .sum()
            })
        });
    }
    let cols = [
        mul(cross(rows[1], rows[2]), 1. / det),
        mul(cross(rows[2], rows[0]), 1. / det),
        mul(cross(rows[0], rows[1]), 1. / det),
    ];
    std::array::from_fn(|i| std::array::from_fn(|j| cols[j][i]))
}
fn mv(a: [[f64; 3]; 3], v: V) -> V {
    a.map(|r| dot(r, v))
}
pub struct Transport {
    quad: Vec<Quadrature>,
    pub mu: Vec<f64>,
    pub kappa: Vec<f64>,
    pub t0: Vec<f64>,
    pub interval: f64,
    pub max_iterations: usize,
    pub max_residual: f64,
    pub viscous_heat_j: f64,
    pub viscous_work_residual_j: f64,
    pub conduction_residual_j: f64,
    root: Vec<f64>,
    diag: Vec<f64>,
    halo: Vec<Q>,
    exchange: ExchangeWorkspace,
}
impl Transport {
    pub fn new(
        m: &Model,
        d: &Domain,
        mu: Vec<f64>,
        kappa: Vec<f64>,
        t0: Vec<f64>,
        interval: f64,
    ) -> Result<Self> {
        let nz = m.background.len();
        ensure!(
            mu.len() == nz
                && kappa.len() == nz
                && t0.len() == nz
                && interval > 0.
                && interval <= 10.,
            "Transport shape/interval"
        );
        ensure!(
            mu.iter()
                .chain(&kappa)
                .chain(&t0)
                .all(|v| v.is_finite() && *v > 0.),
            "Transport coefficients"
        );
        let mut columns = d.owned.iter().copied().collect::<BTreeSet<_>>();
        for &i in &d.owned {
            columns.extend(m.mesh.neighbors(i).into_iter().filter(|&j| d.active[j]));
        }
        let mut quad = vec![];
        for i in columns {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                let position = mul(m.mesh.cells[i].center, RE + m.vertical.centers[k]);
                let mut ends = vec![];
                for j in m.mesh.neighbors(i) {
                    if d.active[j] {
                        ends.push((
                            d.index(j, k, nz),
                            sub(
                                mul(m.mesh.cells[j].center, RE + m.vertical.centers[k]),
                                position,
                            ),
                        ));
                    }
                }
                if k > 0 {
                    ends.push((
                        d.index(i, k - 1, nz),
                        mul(
                            m.mesh.cells[i].center,
                            m.vertical.centers[k - 1] - m.vertical.centers[k],
                        ),
                    ));
                }
                if k + 1 < nz {
                    ends.push((
                        d.index(i, k + 1, nz),
                        mul(
                            m.mesh.cells[i].center,
                            m.vertical.centers[k + 1] - m.vertical.centers[k],
                        ),
                    ));
                }
                let mut mat = [[0.; 3]; 3];
                for &(_, delta) in &ends {
                    let w = 1. / dot(delta, delta);
                    for a in 0..3 {
                        for b in 0..3 {
                            mat[a][b] += w * delta[a] * delta[b];
                        }
                    }
                }
                let inv = inverse(mat);
                let pairs = ends
                    .into_iter()
                    .map(|(j, v)| (j, mv(inv, mul(v, 1. / dot(v, v)))))
                    .collect();
                quad.push(Quadrature {
                    index: n,
                    volume: m.volume(i, k),
                    pairs,
                    layer: k,
                });
            }
        }
        let len = d.needed.len() * nz;
        Ok(Self {
            quad,
            mu,
            kappa,
            t0,
            interval,
            max_iterations: 0,
            max_residual: 0.,
            viscous_heat_j: 0.,
            viscous_work_residual_j: 0.,
            conduction_residual_j: 0.,
            root: vec![0.; len],
            diag: vec![0.; len],
            halo: vec![[0.; 7]; len],
            exchange: d.exchange_workspace(nz),
        })
    }
    fn stresses(&self, v: &[Q], thermal: bool) -> Vec<[[f64; 3]; 3]> {
        self.quad
            .iter()
            .map(|q| {
                let mut g = [[0.; 3]; 3];
                for &(n, c) in &q.pairs {
                    for a in 0..if thermal { 1 } else { 3 } {
                        for b in 0..3 {
                            g[a][b] += (v[n][a] - v[q.index][a]) * c[b];
                        }
                    }
                }
                if thermal {
                    let mut s = [[0.; 3]; 3];
                    s[0] = mul(g[0], self.kappa[q.layer]);
                    s
                } else {
                    let tr = (g[0][0] + g[1][1] + g[2][2]) / 3.;
                    std::array::from_fn(|a| {
                        std::array::from_fn(|b| {
                            self.mu[q.layer]
                                * (g[a][b] + g[b][a] - if a == b { 2. * tr } else { 0. })
                        })
                    })
                }
            })
            .collect()
    }
    fn apply(
        &mut self,
        m: &Model,
        d: &Domain,
        w: &impl Communicator,
        x: &[Q],
        dt: f64,
        thermal: bool,
    ) -> Vec<Q> {
        let nz = m.background.len();
        let nc = if thermal { 1 } else { 3 };
        self.halo.fill([0.; 7]);
        for &i in &d.owned {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                for a in 0..nc {
                    self.halo[n][a] = x[n][a] / self.root[n];
                }
            }
        }
        d.exchange_cached(w, &mut self.halo, nz, &mut self.exchange);
        let stresses = self.stresses(&self.halo, thermal);
        let mut result = x.to_vec();
        for (q, s) in self.quad.iter().zip(stresses) {
            for &(n, c) in &q.pairs {
                let force = if thermal {
                    [dot(s[0], c), 0., 0.]
                } else {
                    mv(s, c)
                };
                for (idx, sign) in [(q.index, -1.), (n, 1.)] {
                    let col = d.needed[idx / nz];
                    if d.owned.binary_search(&col).is_ok() {
                        for a in 0..nc {
                            result[idx][a] += dt * sign * q.volume * force[a] / self.root[idx];
                        }
                    }
                }
            }
        }
        result
    }
    fn inner(
        d: &Domain,
        nz: usize,
        w: &impl CommunicatorCollectives,
        a: &[Q],
        b: &[Q],
        nc: usize,
    ) -> f64 {
        let mut local = 0.;
        for &i in &d.owned {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                for j in 0..nc {
                    local += a[n][j] * b[n][j];
                }
            }
        }
        let mut total = 0.;
        w.all_reduce_into(&local, &mut total, SystemOperation::sum());
        total
    }
    fn solve(
        &mut self,
        m: &Model,
        d: &Domain,
        w: &impl CommunicatorCollectives,
        b: &[Q],
        dt: f64,
        thermal: bool,
    ) -> Result<Vec<Q>> {
        let nz = m.background.len();
        let nc = if thermal { 1 } else { 3 };
        self.diag.fill(1.);
        for q in &self.quad {
            let scale = if thermal {
                self.kappa[q.layer]
            } else {
                4. / 3. * self.mu[q.layer]
            };
            let mut center = [0.; 3];
            for &(n, c) in &q.pairs {
                center = add(center, c);
                self.diag[n] += dt * q.volume * scale * dot(c, c) / (self.root[n] * self.root[n]);
            }
            self.diag[q.index] += dt * q.volume * scale * dot(center, center)
                / (self.root[q.index] * self.root[q.index]);
        }
        let mut x = b.to_vec();
        let ap = self.apply(m, d, w, &x, dt, thermal);
        let mut r = b.to_vec();
        let mut z = b.to_vec();
        for n in 0..b.len() {
            for a in 0..nc {
                r[n][a] -= ap[n][a];
                z[n][a] = r[n][a] / self.diag[n];
            }
        }
        let mut p = z.clone();
        let norm = Self::inner(d, nz, w, b, b, nc).sqrt();
        if norm == 0. {
            return Ok(x);
        }
        let mut rz = Self::inner(d, nz, w, &r, &z, nc);
        let mut iterations = 0;
        while Self::inner(d, nz, w, &r, &r, nc).sqrt() / norm > 1e-9 && iterations < 400 {
            let ap = self.apply(m, d, w, &p, dt, thermal);
            let pa = Self::inner(d, nz, w, &p, &ap, nc);
            ensure!(pa.is_finite() && pa > 0., "Transport operator lost SPD");
            let alpha = rz / pa;
            for &i in &d.owned {
                for k in 0..nz {
                    let n = d.index(i, k, nz);
                    for a in 0..nc {
                        x[n][a] += alpha * p[n][a];
                        r[n][a] -= alpha * ap[n][a];
                        z[n][a] = r[n][a] / self.diag[n];
                    }
                }
            }
            let next = Self::inner(d, nz, w, &r, &z, nc);
            let beta = next / rz;
            rz = next;
            for &i in &d.owned {
                for k in 0..nz {
                    let n = d.index(i, k, nz);
                    for a in 0..nc {
                        p[n][a] = z[n][a] + beta * p[n][a];
                    }
                }
            }
            iterations += 1;
        }
        let ap = self.apply(m, d, w, &x, dt, thermal);
        for n in 0..b.len() {
            for a in 0..nc {
                r[n][a] = b[n][a] - ap[n][a];
            }
        }
        let actual = Self::inner(d, nz, w, &r, &r, nc).sqrt() / norm;
        self.max_iterations = self.max_iterations.max(iterations);
        self.max_residual = self.max_residual.max(actual);
        ensure!(
            actual < 5e-9,
            "Implicit transport failed residual gate: {actual}"
        );
        Ok(x)
    }
    pub fn advance(
        &mut self,
        m: &Model,
        d: &Domain,
        w: &impl CommunicatorCollectives,
        u: &mut [Q],
        dt: f64,
    ) -> Result<()> {
        let nz = m.background.len();
        d.exchange_cached(w, u, nz, &mut self.exchange);
        let mut b = vec![[0.; 7]; u.len()];
        for (slot, &i) in d.needed.iter().enumerate() {
            for k in 0..nz {
                let n = slot * nz + k;
                self.root[n] = (u[n][0] * m.volume(i, k)).sqrt();
                for a in 0..3 {
                    b[n][a] = self.root[n] * u[n][a + 1] / u[n][0];
                }
            }
        }
        let x = self.solve(m, d, w, &b, dt, false)?;
        // Final implicit velocity for stress, and midpoint velocity for paired work.
        for n in 0..u.len() {
            for a in 0..3 {
                self.halo[n][a] = x[n][a] / self.root[n];
            }
        }
        d.exchange_cached(w, &mut self.halo, nz, &mut self.exchange);
        let stresses = self.stresses(&self.halo, false);
        let mut mid = vec![[0.; 7]; u.len()];
        for &i in &d.owned {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                for a in 0..3 {
                    mid[n][a] = 0.5 * (b[n][a] + x[n][a]) / self.root[n];
                }
            }
        }
        d.exchange_cached(w, &mut mid, nz, &mut self.exchange);
        let mut de = vec![0.; u.len()];
        for (q, s) in self.quad.iter().zip(stresses) {
            for &(n, c) in &q.pairs {
                let force = mv(s, c);
                let vm = std::array::from_fn(|a| 0.5 * (mid[n][a] + mid[q.index][a]));
                let flux = dt * q.volume * dot(force, vm);
                de[q.index] += flux;
                de[n] -= flux;
            }
        }
        let mut local = [0.; 3];
        for &i in &d.owned {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                let vol = m.volume(i, k);
                for a in 0..3 {
                    local[0] += 0.5 * (b[n][a] * b[n][a] - x[n][a] * x[n][a]);
                    u[n][a + 1] = u[n][0] * x[n][a] / self.root[n];
                }
                u[n][4] += de[n] / vol;
                local[1] += de[n];
                ensure!(admissible(u[n], m.background[k]), "Viscous work positivity");
            }
        }
        d.exchange_cached(w, u, nz, &mut self.exchange);
        for (slot, &i) in d.needed.iter().enumerate() {
            for k in 0..nz {
                let n = slot * nz + k;
                let a = m.background[k];
                let cv = (u[n][0] - u[n][5] - u[n][6]) * a.cv
                    + u[n][5] * 3. * (1.380649e-23 / (16.043 * 1.66053906660e-27))
                    + u[n][6] * 2.5 * (1.380649e-23 / (31.998 * 1.66053906660e-27));
                let kin =
                    (u[n][1] * u[n][1] + u[n][2] * u[n][2] + u[n][3] * u[n][3]) / (2. * u[n][0]);
                let reference = a.p * a.cv / a.r + (cv - a.rho * a.cv) * self.t0[k];
                self.root[n] = (cv * m.volume(i, k)).sqrt();
                b[n] = [0.; 7];
                b[n][0] = self.root[n] * (u[n][4] - kin - reference) / cv;
            }
        }
        let x = self.solve(m, d, w, &b, dt, true)?;
        for &i in &d.owned {
            for k in 0..nz {
                let n = d.index(i, k, nz);
                let energy = self.root[n] * (x[n][0] - b[n][0]);
                u[n][4] += energy / m.volume(i, k);
                local[2] += energy;
                ensure!(admissible(u[n], m.background[k]), "Conduction positivity");
            }
        }
        let mut total = [0.; 3];
        w.all_reduce_into(&local, &mut total, SystemOperation::sum());
        self.viscous_heat_j += total[0];
        self.viscous_work_residual_j += total[1];
        self.conduction_residual_j += total[2];
        Ok(())
    }
}
