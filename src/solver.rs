use crate::{mesh::*, parallel::Domain, physics::*};
use std::collections::BTreeSet;
pub struct Model {
    pub mesh: Mesh,
    pub background: Vec<Atmos>,
    pub dz: f64, // Legacy nominal spacing; geometry uses vertical.
    pub vertical: crate::vertical::VerticalGrid,
    pub rotation: bool,
    pub sponge: bool,
    pub second_order: bool,
    pub reflect_top: bool,
    pub initial_sigma_m: f64,
    pub initial_surface_pressure_pa: f64,
    pub heat_pulse: bool,
}
#[derive(Clone)]
struct Slopes {
    horizontal: [V; 7],
    vertical: Q,
}
/// Fixed topology and reusable stage buffers; rebuild when mesh/domain/grid changes.
pub struct RhsWorkspace {
    columns: Vec<usize>,
    slopes: Vec<Option<Slopes>>,
    edges: Vec<usize>,
    cell_edges: Vec<[usize; 3]>,
    fluxes: Vec<Q>,
    result: Vec<Q>,
}
impl RhsWorkspace {
    pub fn new(m: &Model, d: &Domain) -> Self {
        let nz = m.background.len();
        let mut columns = d.owned.iter().copied().collect::<BTreeSet<_>>();
        let mut edges = BTreeSet::new();
        for &i in &d.owned {
            columns.extend(m.mesh.neighbors(i));
            edges.extend(m.mesh.cells[i].edges);
        }
        let edges = edges.into_iter().collect::<Vec<_>>();
        let cell_edges = d
            .owned
            .iter()
            .map(|&i| {
                m.mesh.cells[i]
                    .edges
                    .map(|e| edges.binary_search(&e).unwrap())
            })
            .collect();
        Self {
            columns: columns.into_iter().collect(),
            slopes: vec![None; d.needed.len() * nz],
            fluxes: vec![[0.; 7]; edges.len() * nz],
            edges,
            cell_edges,
            result: vec![[0.; 7]; d.needed.len() * nz],
        }
    }
}
impl Model {
    pub fn isothermal(level: usize, nz: usize, dz: f64) -> Self {
        let rt = 287. * 300.;
        let h = rt / 9.81;
        Self {
            mesh: Mesh::new(level),
            background: (0..nz)
                .map(|k| {
                    let rho = (-((k as f64 + 0.5) * dz) / h).exp();
                    Atmos {
                        rho,
                        p: rho * rt,
                        r: 287.,
                        cv: 717.5,
                        g: 9.81,
                    }
                })
                .collect(),
            dz,
            vertical: crate::vertical::VerticalGrid::uniform(nz, dz),
            rotation: false,
            reflect_top: false,
            sponge: false,
            second_order: true,
            initial_sigma_m: 1_000_000.,
            initial_surface_pressure_pa: 1e-7 * 287. * 300.,
            heat_pulse: false,
        }
    }
    pub fn initial(&self, d: &Domain, lamb: bool) -> Vec<Q> {
        let nz = self.background.len();
        // Tonga location, but an idealized weak initial mode, not an eruption fit.
        let lat = (-20.54_f64).to_radians();
        let lon = (-175.38_f64).to_radians();
        let source = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let h = 287. * 300. / 9.81;
        let c2 = 1.4 * 287. * 300.;
        let sigma = self.initial_sigma_m;
        let amp = self.initial_surface_pressure_pa;
        let mut u = vec![[0.; 7]; d.needed.len() * nz];
        for &i in &d.needed {
            let dist = angle(self.mesh.cells[i].center, source) * RE;
            for k in 0..nz {
                let a = self.background[k];
                let mut q = base(a);
                if lamb {
                    let z = self.vertical.centers[k];
                    let vertical = if self.heat_pulse {
                        (-0.5 * ((z - 10_000.) / 5_000.).powi(2)).exp()
                    } else {
                        (-z / (1.4 * h)).exp()
                    };
                    let pp = amp * vertical * (-dist * dist / (2. * sigma * sigma)).exp();
                    if !self.heat_pulse {
                        q[0] += pp / c2;
                    }
                    q[4] += pp * a.cv / a.r;
                }
                u[d.index(i, k, nz)] = q;
            }
        }
        u
    }
    pub fn volume(&self, i: usize, k: usize) -> f64 {
        self.mesh
            .shell_volume(i, self.vertical.faces[k], self.vertical.faces[k + 1])
    }
    pub fn set_vertical(&mut self, grid: crate::vertical::VerticalGrid) {
        let rt = 287. * 300.;
        self.background = grid
            .centers
            .iter()
            .map(|z| {
                let rho = (-z / (rt / 9.81)).exp();
                Atmos {
                    rho,
                    p: rho * rt,
                    r: 287.,
                    cv: 717.5,
                    g: 9.81,
                }
            })
            .collect();
        self.vertical = grid;
    }
    pub fn rhs(&self, d: &Domain, u: &[Q]) -> Vec<Q> {
        let mut work = RhsWorkspace::new(self, d);
        self.rhs_impl(d, u, &mut work, false).to_vec()
    }
    pub fn rhs_cached<'a>(&self, d: &Domain, u: &[Q], work: &'a mut RhsWorkspace) -> &'a mut [Q] {
        self.rhs_impl(d, u, work, true)
    }
    fn rhs_impl<'a>(
        &self,
        d: &Domain,
        u: &[Q],
        work: &'a mut RhsWorkspace,
        cached: bool,
    ) -> &'a mut [Q] {
        let nz = self.background.len();
        assert_eq!(u.len(), work.result.len());
        let columns = &work.columns;
        let slopes = &mut work.slopes;
        if self.second_order {
            for &i in columns {
                let up = self.mesh.cells[i].center;
                let axis = if up[2].abs() < 0.9 {
                    [0., 0., 1.]
                } else {
                    [1., 0., 0.]
                };
                let e = unit(cross(axis, up));
                let n = cross(up, e);
                for k in 0..nz {
                    let radius = RE + self.vertical.centers[k];
                    let p = primitive(u[d.index(i, k, nz)], self.background[k]);
                    let mut a = 0.;
                    let mut b = 0.;
                    let mut c = 0.;
                    let mut bx = [0.; 7];
                    let mut by = [0.; 7];
                    let mut pmin = p;
                    let mut pmax = p;
                    for j in self.mesh.neighbors(i) {
                        let delta = mul(sub(self.mesh.cells[j].center, up), radius);
                        let x = dot(delta, e);
                        let y = dot(delta, n);
                        let weight = 1. / (x * x + y * y);
                        a += weight * x * x;
                        b += weight * x * y;
                        c += weight * y * y;
                        let neighbor = primitive(u[d.index(j, k, nz)], self.background[k]);
                        for f in 0..7 {
                            bx[f] += weight * x * (neighbor[f] - p[f]);
                            by[f] += weight * y * (neighbor[f] - p[f]);
                            pmin[f] = pmin[f].min(neighbor[f]);
                            pmax[f] = pmax[f].max(neighbor[f]);
                        }
                    }
                    let det = a * c - b * b;
                    let mut horizontal = [[0.; 3]; 7];
                    for f in 0..7 {
                        horizontal[f] = add(
                            mul(e, (c * bx[f] - b * by[f]) / det),
                            mul(n, (a * by[f] - b * bx[f]) / det),
                        );
                        let mut theta = 1f64;
                        for j in self.mesh.neighbors(i) {
                            let delta = mul(sub(self.mesh.cells[j].center, up), radius / 2.);
                            let change = dot(horizontal[f], delta);
                            if change > 0. {
                                theta = theta.min((pmax[f] - p[f]) / change);
                            } else if change < 0. {
                                theta = theta.min((pmin[f] - p[f]) / change);
                            }
                        }
                        horizontal[f] = mul(horizontal[f], theta.clamp(0., 1.));
                    }
                    let vertical = if k > 0 && k + 1 < nz {
                        let below = primitive(u[d.index(i, k - 1, nz)], self.background[k - 1]);
                        let above = primitive(u[d.index(i, k + 1, nz)], self.background[k + 1]);
                        std::array::from_fn(|f| {
                            let dl = self.vertical.centers[k] - self.vertical.centers[k - 1];
                            let dr = self.vertical.centers[k + 1] - self.vertical.centers[k];
                            if dl == self.vertical.widths[k] && dr == dl {
                                return mc(p[f] - below[f], above[f] - p[f]);
                            }
                            let l = (p[f] - below[f]) / dl;
                            let r = (above[f] - p[f]) / dr;
                            if l * r <= 0. {
                                0.
                            } else {
                                l.signum()
                                    * (2. * l.abs())
                                        .min(2. * r.abs())
                                        .min(((dl * l + dr * r) / (dl + dr)).abs())
                                    * self.vertical.widths[k]
                            }
                        })
                    } else {
                        [0.; 7]
                    };
                    slopes[d.index(i, k, nz)] = Some(Slopes {
                        horizontal,
                        vertical,
                    });
                }
            }
        }
        let reconstruct = |i: usize, k: usize, delta: V, vertical: f64| {
            let q = u[d.index(i, k, nz)];
            if !self.second_order || q == base(self.background[k]) {
                return q;
            }
            let p = primitive(q, self.background[k]);
            let slope = slopes[d.index(i, k, nz)]
                .as_ref()
                .expect("Missing face slope");
            let face = std::array::from_fn(|f| {
                p[f] + dot(slope.horizontal[f], delta) + vertical * slope.vertical[f]
            });
            let trial = conserved(face, self.background[k]);
            if admissible(trial, self.background[k]) {
                trial
            } else {
                q
            }
        };
        if cached {
            for (slot, &eid) in work.edges.iter().enumerate() {
                let edge = &self.mesh.edges[eid];
                for k in 0..nz {
                    let radius = RE + self.vertical.centers[k];
                    let delta = mul(
                        sub(
                            self.mesh.cells[edge.right].center,
                            self.mesh.cells[edge.left].center,
                        ),
                        radius / 2.,
                    );
                    work.fluxes[slot * nz + k] = hll(
                        reconstruct(edge.left, k, delta, 0.),
                        reconstruct(edge.right, k, mul(delta, -1.), 0.),
                        self.background[k],
                        self.background[k],
                        edge.outward,
                    );
                }
            }
        }
        let result = &mut work.result;
        result.fill([0.; 7]);
        for (owned_slot, &i) in d.owned.iter().enumerate() {
            let cell = &self.mesh.cells[i];
            let up = cell.center;
            for k in 0..nz {
                let a = self.background[k];
                let q = u[d.index(i, k, nz)];
                let lo = RE + self.vertical.faces[k];
                let hi = RE + self.vertical.faces[k + 1];
                let radius = (lo + hi) / 2.;
                let volume = self.volume(i, k);
                let mut rate = [0.; 7];
                for (edge_slot, &eid) in cell.edges.iter().enumerate() {
                    let edge = &self.mesh.edges[eid];
                    let delta = mul(
                        sub(
                            self.mesh.cells[edge.right].center,
                            self.mesh.cells[edge.left].center,
                        ),
                        radius / 2.,
                    );
                    let flux = if cached {
                        work.fluxes[work.cell_edges[owned_slot][edge_slot] * nz + k]
                    } else {
                        let left = reconstruct(edge.left, k, delta, 0.);
                        let right = reconstruct(edge.right, k, mul(delta, -1.), 0.);
                        hll(left, right, a, a, edge.outward)
                    };
                    let area = 0.5 * (hi * hi - lo * lo) * edge.angle;
                    let sign = if edge.left == i { 1. } else { -1. };
                    for f in 0..7 {
                        rate[f] -= sign * area * flux[f] / volume;
                    }
                }
                let normal = unit(cell.vector_area);
                let va = norm(cell.vector_area);
                let high = reconstruct(i, k, [0.; 3], 0.5);
                let (right, ar) = if k + 1 < nz {
                    (reconstruct(i, k + 1, [0.; 3], -0.5), self.background[k + 1])
                } else if self.reflect_top {
                    let mut mirror = high;
                    let radial = dot([high[1], high[2], high[3]], normal);
                    for t in 0..3 {
                        mirror[t + 1] -= 2. * radial * normal[t];
                    }
                    (mirror, a)
                } else {
                    (base(a), a)
                };
                let top_fraction = if k + 1 < nz {
                    self.vertical.widths[k]
                        / (self.vertical.widths[k] + self.vertical.widths[k + 1])
                } else {
                    0.5
                };
                let top = hll_at(high, right, a, ar, normal, top_fraction);
                let low = reconstruct(i, k, [0.; 3], -0.5);
                let (left, al) = if k > 0 {
                    (reconstruct(i, k - 1, [0.; 3], 0.5), self.background[k - 1])
                } else {
                    let mut mirror = low;
                    let radial = dot([low[1], low[2], low[3]], normal);
                    for t in 0..3 {
                        mirror[t + 1] -= 2. * radial * normal[t];
                    }
                    (mirror, a)
                };
                let bottom_fraction = if k > 0 {
                    self.vertical.widths[k - 1]
                        / (self.vertical.widths[k - 1] + self.vertical.widths[k])
                } else {
                    0.5
                };
                let bottom = hll_at(left, low, al, a, normal, bottom_fraction);
                for f in 0..7 {
                    rate[f] -= va * (hi * hi * top[f] - lo * lo * bottom[f]) / volume;
                }
                for t in 0..3 {
                    rate[t + 1] -= (q[0] - a.rho) * a.g * up[t];
                }
                rate[4] -= dot([q[1], q[2], q[3]], up) * a.g;
                if self.rotation {
                    let coriolis = mul(cross([0., 0., 7.292115e-5], [q[1], q[2], q[3]]), -2.);
                    for t in 0..3 {
                        rate[t + 1] += coriolis[t];
                    }
                }
                if self.sponge {
                    let height = self.vertical.centers[k];
                    let top_height = *self.vertical.faces.last().unwrap();
                    let start = 0.75 * top_height;
                    let fraction = ((height - start) / (top_height - start)).clamp(0., 1.);
                    let damping = fraction * fraction / 100.;
                    let b = base(a);
                    for f in 0..7 {
                        rate[f] -= damping * (q[f] - b[f]);
                    }
                }
                result[d.index(i, k, nz)] = rate;
            }
        }
        result
    }
    pub fn local_dt(&self, d: &Domain, u: &[Q]) -> f64 {
        let nz = self.background.len();
        let mut dt = f64::INFINITY;
        for &i in &d.owned {
            for k in 0..nz {
                let q = u[d.index(i, k, nz)];
                let c = thermo(q, self.background[k]).1;
                let speed = norm([q[1], q[2], q[3]]) / q[0];
                let lo = RE + self.vertical.faces[k];
                let hi = RE + self.vertical.faces[k + 1];
                let side = self.mesh.cells[i]
                    .edges
                    .iter()
                    .map(|&e| 0.5 * (hi * hi - lo * lo) * self.mesh.edges[e].angle)
                    .sum::<f64>();
                let vertical = (hi * hi + lo * lo) * norm(self.mesh.cells[i].vector_area);
                dt = dt.min(0.3 * self.volume(i, k) / ((c + speed) * (side + vertical)));
            }
        }
        dt
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hydrostatic_equilibrium() {
        let mut m = Model::isothermal(1, 6, 1000.);
        m.reflect_top = true;
        let d = Domain::new(&m.mesh, 0, 1);
        let u = m.initial(&d, false);
        let r = m.rhs(&d, &u);
        assert!(r.iter().flatten().all(|x| x.abs() < 1e-12));
    }
    #[test]
    fn closed_horizontal_flux_conserves_mass() {
        let mut m = Model::isothermal(1, 6, 1000.);
        m.reflect_top = true;
        let d = Domain::new(&m.mesh, 0, 1);
        let mut u = m.initial(&d, true);
        for &i in &d.owned {
            let tangent = cross([0., 0., 1.], m.mesh.cells[i].center);
            for k in 0..6 {
                let s = d.index(i, k, 6);
                let mut p = primitive(u[s], m.background[k]);
                for f in 0..3 {
                    p[f + 1] = tangent[f] * 0.01;
                }
                u[s] = conserved(p, m.background[k]);
            }
        }
        let r = m.rhs(&d, &u);
        let mut total = 0.;
        let mut abs = 0.;
        for &i in &d.owned {
            for k in 0..6 {
                let value = r[d.index(i, k, 6)][0] * m.mesh.volume(i, k, 1000.);
                total += value;
                abs += value.abs();
            }
        }
        assert!(total.abs() / abs.max(1.) < 1e-12);
    }
}
