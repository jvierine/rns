//! Implicit Newtonian deviatoric stress and Fourier conduction on actual prisms.
//! A least-squares Cartesian gradient and its exact volume adjoint form an SPD
//! operator, with affine-exact face-jump stabilization against checkerboard modes.
#[cfg(all(feature="native-metal",target_os="macos"))]
#[path="metal_transport.rs"] mod metal_transport;
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
struct Penalty {
    entries: Vec<(usize,f64)>,
    weight: f64,
    layer: usize,
}
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
    #[cfg(all(feature="native-metal",target_os="macos"))]
    metal:Option<metal_transport::MetalTransport>,
    pub metal_refinement_passes:usize,
    pub relative_tolerance:f64,
    pub metal_inner_max_residual:f64,
    quad: Vec<Quadrature>,
    penalties: Vec<Penalty>,
    incidence: Vec<Vec<(usize,V)>>,
    penalty_offsets:Vec<usize>,
    penalty_slots:Vec<u32>,
    penalty_coefficients:Vec<f64>,
    owned: Vec<bool>,
    threads: usize,
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
    stiffness_diag_viscous: Vec<f64>,
    stiffness_diag_thermal: Vec<f64>,
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
        // Penalize only the jump unexplained by the reconstructed gradient.
        // This vanishes for affine fields and closes the centered-gradient
        // checkerboard null space. The exact adjoint preserves SPD and budgets.
        let mut penalties=vec![];
        for q in &quad {
            let column=d.needed[q.index/nz];
            let position=mul(m.mesh.cells[column].center,RE+m.vertical.centers[q.layer]);
            for &(j,_) in &q.pairs {
                let jc=d.needed[j/nz];
                let delta=sub(mul(m.mesh.cells[jc].center,RE+m.vertical.centers[j%nz]),position);
                let mut coeff=std::collections::BTreeMap::<usize,f64>::new();
                *coeff.entry(j).or_default()+=1.;
                *coeff.entry(q.index).or_default()-=1.;
                for &(n,c) in &q.pairs {let b=dot(c,delta);*coeff.entry(n).or_default()-=b;*coeff.entry(q.index).or_default()+=b;}
                let entries:Vec<_>=coeff.into_iter().filter(|(_,b)|b.abs()>1e-15).collect();
                // A fully affine-exact boundary stencil can have zero residual.
                if !entries.is_empty(){penalties.push(Penalty{entries,weight:q.volume/(2.*dot(delta,delta)),layer:q.layer});}
            }
        }
        let len = d.needed.len() * nz;
        let mut incidence=vec![vec![];len];
        for (slot,q) in quad.iter().enumerate(){for &(n,c) in &q.pairs{incidence[q.index].push((slot,mul(c,-q.volume)));incidence[n].push((slot,mul(c,q.volume)));}}
        let owned:Vec<bool>=(0..len).map(|n|d.owned.binary_search(&d.needed[n/nz]).is_ok()).collect();
        let mut penalty_offsets=vec![0usize;len+1];
        for p in &penalties{for &(n,_)in &p.entries{if owned[n]{penalty_offsets[n+1]+=1;}}}
        for n in 0..len{penalty_offsets[n+1]+=penalty_offsets[n];}
        let mut penalty_slots=vec![0u32;penalty_offsets[len]];let mut penalty_coefficients=vec![0.;penalty_offsets[len]];let mut cursor=penalty_offsets[..len].to_vec();
        ensure!(penalties.len()<=u32::MAX as usize,"Penalty index overflow");
        for (slot,p)in penalties.iter().enumerate(){for &(n,b)in &p.entries{if owned[n]{let i=cursor[n];penalty_slots[i]=slot as u32;penalty_coefficients[i]=b;cursor[n]+=1;}}}
        let threads=std::env::var("RNS_TRANSPORT_THREADS").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or_else(||if d.size==1 {std::thread::available_parallelism().map(|n|n.get().min(8)).unwrap_or(1)}else{1}).max(1);
        let stiffness_diagonal=|thermal:bool| {
            let mut diagonal=vec![0.;len];
            for q in &quad {
                let scale=if thermal{kappa[q.layer]}else{4./3.*mu[q.layer]};
                let mut center=[0.;3];
                for &(n,c) in &q.pairs {center=add(center,c);diagonal[n]+=q.volume*scale*dot(c,c);}
                diagonal[q.index]+=q.volume*scale*dot(center,center);
            }
            for p in &penalties {let coefficient=p.weight*if thermal{kappa[p.layer]}else{mu[p.layer]};for &(n,b)in &p.entries{diagonal[n]+=coefficient*b*b;}}
            diagonal
        };
        let stiffness_diag_viscous=stiffness_diagonal(false);
        let stiffness_diag_thermal=stiffness_diagonal(true);
        Ok(Self {
            #[cfg(all(feature="native-metal",target_os="macos"))]
            metal:None,
            metal_refinement_passes:0,metal_inner_max_residual:0.,relative_tolerance:1e-9,
            stiffness_diag_viscous,stiffness_diag_thermal,
            quad,
            penalties,
            incidence,owned,threads,penalty_offsets,penalty_slots,penalty_coefficients,
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
    pub fn enable_metal(&mut self,m:&Model,d:&Domain)->Result<()> {
        #[cfg(all(feature="native-metal",target_os="macos"))]
        {self.metal=Some(metal_transport::MetalTransport::new(&self.quad,&self.penalties,&self.incidence,&self.owned,&self.mu,&self.kappa,m,d)?);return Ok(());}
        #[cfg(not(all(feature="native-metal",target_os="macos")))]
        {anyhow::bail!("Native Metal transport unavailable");}
    }
    fn apply_for_cg(&mut self,m:&Model,d:&Domain,w:&impl Communicator,x:&[Q],dt:f64,thermal:bool,gpu:bool)->Result<Vec<Q>> {
        if gpu {
            #[cfg(all(feature="native-metal",target_os="macos"))]
            {return self.metal.as_mut().ok_or_else(||anyhow::anyhow!("Missing transport GPU"))?.apply(x,&self.root,dt,thermal);}
            #[cfg(not(all(feature="native-metal",target_os="macos")))]
            {anyhow::bail!("GPU transport disabled");}
        }
        Ok(self.apply(m,d,w,x,dt,thermal))
    }
    fn solve(&mut self,m:&Model,d:&Domain,w:&impl CommunicatorCollectives,b:&[Q],dt:f64,thermal:bool)->Result<Vec<Q>> {
        #[cfg(not(all(feature="native-metal",target_os="macos")))]
        let gpu=false;
        #[cfg(all(feature="native-metal",target_os="macos"))]
        let gpu=self.metal.is_some();
        ensure!(self.relative_tolerance.is_finite() && (1e-12..=1e-5).contains(&self.relative_tolerance),"Invalid transport relative tolerance");
        if !gpu{return self.solve_cg(m,d,w,b,dt,thermal,false,self.relative_tolerance);}
        ensure!(d.size==1,"Metal transport currently requires one MPI rank");
        let nz=m.background.len();let nc=if thermal{1}else{3};
        let norm=Self::inner(d,nz,w,b,b,nc).sqrt();if norm==0.{return Ok(b.to_vec());}
        // Begin from the unchanged physical state. In a nearly equilibrated column
        // the transport correction can be far smaller than float32 state ULPs.
        // Form b-Ax in float64 and send only that correction to Metal.
        let mut x=b.to_vec();
        for pass in 0..6 {
            // Residual is ALWAYS evaluated using the original float64 operator.
            let ax=self.apply(m,d,w,&x,dt,thermal);let mut r=b.to_vec();
            for n in 0..r.len(){for a in 0..nc{r[n][a]-=ax[n][a];}}
            let residual=Self::inner(d,nz,w,&r,&r,nc).sqrt()/norm;
            if residual<self.relative_tolerance{self.max_residual=self.max_residual.max(residual);self.metal_refinement_passes=self.metal_refinement_passes.max(pass);return Ok(x);}
            // Avoid oversolving a small correction; final acceptance remains 1e-9
            // with the original float64 operator on every iteration.
            let inner_tolerance=(0.25*self.relative_tolerance/residual).clamp(1e-3,0.05);
            let correction=self.solve_cg(m,d,w,&r,dt,thermal,true,inner_tolerance)?;
            for &i in &d.owned{for k in 0..nz{let n=d.index(i,k,nz);for a in 0..nc{x[n][a]+=correction[n][a];}}}
        }
        anyhow::bail!("Metal transport float64 residual correction did not converge; no fallback")
    }
    fn stresses(&self, v: &[Q], thermal: bool) -> Vec<[[f64; 3]; 3]> {
        let calculate=|q:&Quadrature| {
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
        };
        let mut result=vec![[[0.;3];3];self.quad.len()];
        if self.threads>1 && self.quad.len()>100_000 {
            let chunk=self.quad.len().div_ceil(self.threads);
            std::thread::scope(|scope|{for (out,quad) in result.chunks_mut(chunk).zip(self.quad.chunks(chunk)){let calc=&calculate;scope.spawn(move ||{for (out,q) in out.iter_mut().zip(quad){*out=calc(q);}});}});
        }else{for (out,q) in result.iter_mut().zip(&self.quad){*out=calculate(q);}}
        result
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
        let calculate=|n:usize,out:&mut Q| {
            if !self.owned[n]{return;}
            for &(slot,c) in &self.incidence[n]{let s=stresses[slot];let force=if thermal{[dot(s[0],c),0.,0.]}else{mv(s,c)};for a in 0..nc{out[a]+=dt*force[a]/self.root[n];}}
        };
        if self.threads>1 && result.len()>100_000 {
            let chunk=result.len().div_ceil(self.threads);
            std::thread::scope(|scope|{for (slot,out) in result.chunks_mut(chunk).enumerate(){let calc=&calculate;scope.spawn(move ||{for (j,q) in out.iter_mut().enumerate(){calc(slot*chunk+j,q);}});}});
        }else{for (n,out) in result.iter_mut().enumerate(){calculate(n,out);}}
        if self.threads>1 && std::env::var("RNS_TRANSPORT_SERIAL_PENALTY").as_deref()!=Ok("1") {
            let mut residuals=vec![[0.;3];self.penalties.len()];let chunk=self.penalties.len().div_ceil(self.threads).max(1);
            std::thread::scope(|scope|{for(out,penalties)in residuals.chunks_mut(chunk).zip(self.penalties.chunks(chunk)){let halo=&self.halo;scope.spawn(move||{for(out,p)in out.iter_mut().zip(penalties){*out=std::array::from_fn(|a|if a<nc{p.entries.iter().map(|&(n,b)|b*halo[n][a]).sum()}else{0.});}});}});
            let gather=|n:usize,out:&mut Q|{for j in self.penalty_offsets[n]..self.penalty_offsets[n+1]{let slot=self.penalty_slots[j]as usize;let p=&self.penalties[slot];let coefficient=p.weight*if thermal{self.kappa[p.layer]}else{self.mu[p.layer]};let b=self.penalty_coefficients[j];for a in 0..nc{out[a]+=dt*coefficient*b*residuals[slot][a]/self.root[n];}}};
            let chunk=result.len().div_ceil(self.threads).max(1);
            std::thread::scope(|scope|{for(slot,out)in result.chunks_mut(chunk).enumerate(){let gather=&gather;scope.spawn(move||{for(j,q)in out.iter_mut().enumerate(){gather(slot*chunk+j,q);}});}});
        }else {
        for p in &self.penalties {
            let coefficient=p.weight*if thermal{self.kappa[p.layer]}else{self.mu[p.layer]};
            let residual:V=std::array::from_fn(|a|if a<nc{p.entries.iter().map(|&(n,b)|b*self.halo[n][a]).sum()}else{0.});
            for &(n,b) in &p.entries {if self.owned[n]{for a in 0..nc{result[n][a]+=dt*coefficient*b*residual[a]/self.root[n];}}}
        }        }

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
    fn solve_cg(
        &mut self,
        m: &Model,
        d: &Domain,
        w: &impl CommunicatorCollectives,
        b: &[Q],
        dt: f64,
        thermal: bool,
        gpu:bool,
        tolerance:f64,
    ) -> Result<Vec<Q>> {
        let nz = m.background.len();
        let nc = if thermal { 1 } else { 3 };
        let norm = Self::inner(d,nz,w,b,b,nc).sqrt();
        // Exact zero right-hand side needs neither CG nor a matrix application.
        if norm==0. {return Ok(b.to_vec());}
        let stiffness=if thermal{&self.stiffness_diag_thermal}else{&self.stiffness_diag_viscous};
        for n in 0..self.diag.len(){self.diag[n]=1.+dt*stiffness[n]/(self.root[n]*self.root[n]);}
        if gpu {
            #[cfg(all(feature="native-metal",target_os="macos"))]
            {let(x,iterations,residual)=self.metal.as_mut().unwrap().solve(b,&self.root,&self.diag,dt,thermal,tolerance)?;self.max_iterations=self.max_iterations.max(iterations);self.metal_inner_max_residual=self.metal_inner_max_residual.max(residual);return Ok(x);}
        }
        let mut x = b.to_vec();
        let ap = self.apply_for_cg(m, d, w, &x, dt, thermal,gpu)?;
        let mut r = b.to_vec();
        let mut z = b.to_vec();
        for n in 0..b.len() {
            for a in 0..nc {
                r[n][a] -= ap[n][a];
                z[n][a] = r[n][a] / self.diag[n];
            }
        }
        let mut p = z.clone();
        let mut rz = Self::inner(d, nz, w, &r, &z, nc);
        let mut iterations = 0;
        while Self::inner(d, nz, w, &r, &r, nc).sqrt() / norm > tolerance && iterations < 400 {
            let ap = self.apply_for_cg(m, d, w, &p, dt, thermal,gpu)?;
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
        let ap = self.apply_for_cg(m, d, w, &x, dt, thermal,gpu)?;
        for n in 0..b.len() {
            for a in 0..nc {
                r[n][a] = b[n][a] - ap[n][a];
            }
        }
        let actual = Self::inner(d, nz, w, &r, &r, nc).sqrt() / norm;
        self.max_iterations = self.max_iterations.max(iterations);
        if gpu{self.metal_inner_max_residual=self.metal_inner_max_residual.max(actual);}else{self.max_residual=self.max_residual.max(actual);}
        ensure!(
            actual < if gpu{0.1}else{5.*tolerance},
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
        let mut penalty_forces=vec![[0.;3];self.penalties.len()];
        let chunk=self.penalties.len().div_ceil(self.threads).max(1);
        std::thread::scope(|scope|{for (out,penalties)in penalty_forces.chunks_mut(chunk).zip(self.penalties.chunks(chunk)){let halo=&self.halo;let mu=&self.mu;scope.spawn(move||{for(out,p)in out.iter_mut().zip(penalties){*out=std::array::from_fn(|a|p.weight*mu[p.layer]*p.entries.iter().map(|&(n,b)|b*halo[n][a]).sum::<f64>());}});}});
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
        if self.threads>1 && std::env::var("RNS_TRANSPORT_SERIAL_WORK").as_deref()!=Ok("1") {
            let gather=|n:usize,out:&mut f64|{
                if !self.owned[n]{return;}
                let mut previous=usize::MAX;
                for &(slot,_)in &self.incidence[n]{let q=&self.quad[slot];let stress=stresses[slot];
                    if q.index==n {
                        if previous==slot{continue;}previous=slot;
                        for &(other,c)in &q.pairs{let force=mv(stress,c);let vm=std::array::from_fn(|a|0.5*(mid[other][a]+mid[n][a]));*out+=dt*q.volume*dot(force,vm);}
                    }else{let c=q.pairs.iter().find(|&&(other,_)|other==n).unwrap().1;let force=mv(stress,c);let vm=std::array::from_fn(|a|0.5*(mid[n][a]+mid[q.index][a]));*out-=dt*q.volume*dot(force,vm);}
                }
                for j in self.penalty_offsets[n]..self.penalty_offsets[n+1]{let slot=self.penalty_slots[j]as usize;let p=&self.penalties[slot];let force=penalty_forces[slot];let center=p.entries[0].0;
                    if n==center{for &(other,b)in &p.entries{if other!=center{let vm=std::array::from_fn(|a|0.5*(mid[other][a]+mid[center][a]));*out+=dt*b*dot(force,vm);}}}
                    else{let b=self.penalty_coefficients[j];let vm=std::array::from_fn(|a|0.5*(mid[n][a]+mid[center][a]));*out-=dt*b*dot(force,vm);}
                }
            };
            let chunk=de.len().div_ceil(self.threads).max(1);std::thread::scope(|scope|{for(slot,out)in de.chunks_mut(chunk).enumerate(){let gather=&gather;scope.spawn(move||{for(j,e)in out.iter_mut().enumerate(){gather(slot*chunk+j,e);}});}});
        }else{
        for (q, s) in self.quad.iter().zip(stresses) {
            for &(n, c) in &q.pairs {
                let force = mv(s, c);
                let vm = std::array::from_fn(|a| 0.5 * (mid[n][a] + mid[q.index][a]));
                let flux = dt * q.volume * dot(force, vm);
                de[q.index] += flux;
                de[n] -= flux;
            }
        }
        for (p,force) in self.penalties.iter().zip(penalty_forces) {
            let center=p.entries[0].0;
            for &(n,b) in &p.entries {if n!=center{let vm=std::array::from_fn(|a|0.5*(mid[n][a]+mid[center][a]));let flux=dt*b*dot(force,vm);de[center]+=flux;de[n]-=flux;}}
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
                ensure!(admissible_with_tracer_roundoff(u[n], m.background[k],m.tracer_roundoff_fraction), "Viscous work positivity");
            }
        }
        d.exchange_cached(w, u, nz, &mut self.exchange);
        for (slot, &i) in d.needed.iter().enumerate() {
            for k in 0..nz {
                let n = slot * nz + k;
                let a = m.at(i,k);
                let cv = (u[n][0] - u[n][5] - u[n][6]) * a.cv
                    + u[n][5] * 3. * (1.380649e-23 / (16.043 * 1.66053906660e-27))
                    + u[n][6] * 2.5 * (1.380649e-23 / (31.998 * 1.66053906660e-27));
                let kin =
                    (u[n][1] * u[n][1] + u[n][2] * u[n][2] + u[n][3] * u[n][3]) / (2. * u[n][0]);
                let t_ref=if m.column_background.is_some(){a.p/(a.rho*a.r)}else{self.t0[k]};
                let reference = a.p * a.cv / a.r + (cv - a.rho * a.cv) * t_ref;
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
                ensure!(admissible_with_tracer_roundoff(u[n], m.background[k],m.tracer_roundoff_fraction), "Conduction positivity");
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
