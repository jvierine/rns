//! Finite-window incremental analysis update (IAU), not continuous relaxation.
//! Bloom et al. (1996), MWR 124, 1256–1271. Smooth raised-cosine window is
//! our explicit adaptation; dynamics remain active while increments are applied.
use crate::{era5::Era5,mesh::*,parallel::Domain,physics::*,solver::Model};
use anyhow::{ensure,Result};
pub struct AnalysisUpdate {pub increment:Vec<Q>,pub duration:f64,pub lower:f64,pub upper:f64}
impl AnalysisUpdate {
 pub fn new(m:&Model,d:&Domain,e:&Era5,duration:f64,lower:f64,upper:f64)->Result<Self>{
  ensure!(duration.is_finite()&&duration>0.&&lower>=0.&&upper>lower&&upper<=50000.,"Invalid IAU window/altitude: forcing must stop by 50 km");
  let nz=m.background.len();let reference=e.hydrostatic_columns(m);let mut increment=vec![[0.;7];d.needed.len()*nz];
  for &i in &d.needed {let up=m.mesh.cells[i].center;let lat=up[2].asin().to_degrees();let lon=up[1].atan2(up[0]);let east=[-lon.sin(),lon.cos(),0.];let north=cross(up,east);
   for k in 0..nz {let z=m.vertical.centers[k];let weight=Self::height_weight(z,lower,upper);if weight==0.{continue;}
    let a=reference[i*nz+k];let wind=e.sample(z.min(30000.),lat,lon.to_degrees());let fade=(1.-(z-30000.)/20000.).clamp(0.,1.);let v=mul(add(mul(east,wind[2]),mul(north,wind[3])),fade);
    let target=conserved([1.,v[0],v[1],v[2],1.,0.,0.],a);let initial=base(m.at(i,k));let n=d.index(i,k,nz);
    increment[n]=std::array::from_fn(|f|weight*(target[f]-initial[f]));
   }
  }
  Ok(Self{increment,duration,lower,upper})
 }
 pub fn height_weight(z:f64,lower:f64,upper:f64)->f64 {let x=((z-lower)/(upper-lower)).clamp(0.,1.);0.5*(1.+(std::f64::consts::PI*x).cos())}
 /// Exact integrated normalized raised-cosine window. Independent of timestep.
 pub fn fraction(&self,t:f64)->f64 {let x=(t/self.duration).clamp(0.,1.);if x==0.||x==1.{return x;}x-(std::f64::consts::TAU*x).sin()/std::f64::consts::TAU}
 pub fn rate(&self,t:f64,dt:f64)->f64 {(self.fraction(t+dt)-self.fraction(t))/dt}
 pub fn integrals(&self,m:&Model,d:&Domain)->[f64;8]{let nz=m.background.len();let mut out=[0.;8];for &i in &d.owned{for k in 0..nz{let q=self.increment[d.index(i,k,nz)];let vol=m.volume(i,k);for f in 0..7{out[f]+=q[f]*vol;}let phi=m.background[0].g*crate::mesh::RE.powi(2)*(1./crate::mesh::RE-1./(crate::mesh::RE+m.vertical.centers[k]));out[7]+=q[0]*phi*vol;}}out}
}
