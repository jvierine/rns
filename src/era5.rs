//! Dry, flat-ground ERA5 initialization; pressure velocity converted by the exporter.
use crate::{mesh::*,parallel::Domain,physics::*,solver::Model};
use anyhow::{Result,ensure};
pub struct Era5 {pub epoch:f64,z:Vec<f64>,lat:Vec<f64>,lon:Vec<f64>,fields:Vec<Vec<f64>>}
fn read(f:&hdf5::File,n:&str)->Result<Vec<f64>>{Ok(f.dataset(n)?.read_raw::<f64>()?)}
fn bracket(v:&[f64],x:f64)->(usize,f64){let x=x.clamp(v[0],v[v.len()-1]);let k=v.partition_point(|q|*q<=x).saturating_sub(1).min(v.len()-2);(k,(x-v[k])/(v[k+1]-v[k]))}
impl Era5 {
 /// Build column references from ERA5 lower-atmosphere temperature joined to
 /// MSIS aloft, integrating d ln(p)/dz=-g/(R T). Each column starts at the
 /// ERA5 pressure at the first centre. References contain no wind; callers
 /// may add ERA5 winds, whose dynamical balance is a separate constraint.
 /// Inverse temperature is joined through 50-120 km so the upper pressure
 /// matches a common MSIS reference while preserving hydrostatic balance.
 pub fn hydrostatic_columns(&self,m:&Model)->Vec<crate::physics::Atmos>{
  let nz=m.background.len();let mut out=Vec::with_capacity(m.mesh.cells.len()*nz);let upper=m.vertical.centers.iter().position(|z|*z>=120000.).unwrap_or(nz-1);
  for cell in &m.mesh.cells{let lat=cell.center[2].asin().to_degrees();let lon=cell.center[1].atan2(cell.center[0]).to_degrees();let anchor=self.sample(30000.,lat,lon);
   let mut inverse=Vec::with_capacity(nz);let mut shape=Vec::with_capacity(nz);
   for k in 0..nz{let z=m.vertical.centers[k];let a=m.background[k];let t=if z<=30000.{self.sample(z,lat,lon)[1]}else{let w=((z-30000.)/20000.).clamp(0.,1.);(1.-w)*anchor[1]+w*a.p/(a.rho*a.r)};inverse.push(a.g/(a.r*t));let x=((z-50000.)/70000.).clamp(0.,1.);shape.push(6.*x*(1.-x));}
   let first=self.sample(m.vertical.centers[0],lat,lon)[0];let mut integral=0.;let mut weight=0.;for k in 1..=upper{let dz=m.vertical.centers[k]-m.vertical.centers[k-1];integral+=0.5*(inverse[k]+inverse[k-1])*dz;weight+=0.5*(shape[k]+shape[k-1])*dz;}
   if m.vertical.centers[upper]<120000.{weight=0.;}
   // Match a common MSIS upper pressure without imposing a force imbalance:
   // adjust inverse temperature smoothly in 50-120 km, then integrate again.
   let adjustment=if weight>0.{(first.ln()-integral-m.background[upper].p.ln())/weight}else{0.};for k in 0..nz{inverse[k]+=adjustment*shape[k];assert!(inverse[k].is_finite()&&inverse[k]>0.,"Invalid hydrostatic transition temperature");}
   let mut pressure=first;for k in 0..nz{let a=m.background[k];if k>0{pressure*=(-0.5*(inverse[k]+inverse[k-1])*(m.vertical.centers[k]-m.vertical.centers[k-1])).exp();}if k==upper && weight>0.{pressure=m.background[k].p;}let rt=a.g/inverse[k];out.push(Atmos{rho:pressure/rt,p:pressure,..a});}
  }out
 }
 pub fn initialize_hydrostatic_columns(&self,m:&Model,d:&Domain,winds:bool)->Vec<Q>{let nz=m.background.len();let mut out=vec![[0.;7];d.needed.len()*nz];for &i in &d.needed{let up=m.mesh.cells[i].center;let lat=up[2].asin().to_degrees();let lon=up[1].atan2(up[0]);let east=[-lon.sin(),lon.cos(),0.];let north=cross(up,east);for k in 0..nz{let a=m.at(i,k);let z=m.vertical.centers[k];let wind=self.sample(z.min(30000.),lat,lon.to_degrees());let scale=if winds{(1.-(z-30000.)/20000.).clamp(0.,1.)}else{0.};let velocity=mul(add(mul(east,wind[2]),mul(north,wind[3])),scale);out[d.index(i,k,nz)]=if velocity==[0.;3]{base(a)}else{conserved([1.,velocity[0],velocity[1],velocity[2],1.,0.,0.],a)};}}out}
 pub fn load(path:&str)->Result<Self>{let f=hdf5::File::open(path)?;let z=read(&f,"altitude_m")?;let lat=read(&f,"latitude_deg")?;let lon=read(&f,"longitude_deg")?;for v in [&z,&lat,&lon]{ensure!(v.len()>1 && v.iter().all(|v|v.is_finite()) && v.windows(2).all(|w|w[1]>w[0]),"ERA5 coordinates");}
 let fields=["pressure_pa","temperature_k","east_m_s","north_m_s","up_m_s"].map(|n|read(&f,&format!("initial/{n}"))).into_iter().collect::<Result<Vec<_>>>()?;let count=z.len()*lat.len()*lon.len();ensure!(fields.iter().all(|v|v.len()==count && v.iter().all(|x|x.is_finite())) && fields[..2].iter().all(|v|v.iter().all(|x|*x>0.)),"ERA5 fields");Ok(Self{epoch:f.attr("reference_unix_s")?.read_scalar()?,z,lat,lon,fields})}
 pub fn sample(&self,height:f64,latitude:f64,longitude:f64)->[f64;5]{let(k,wz)=bracket(&self.z,height);let(j,wy)=bracket(&self.lat,latitude);let lon=longitude.rem_euclid(360.);let a=self.lon.partition_point(|x|*x<=lon).saturating_sub(1);let b=(a+1)%self.lon.len();let width=if b==0{360.-self.lon[a]+self.lon[0]}else{self.lon[b]-self.lon[a]};let wx=(lon-self.lon[a])/width;
 std::array::from_fn(|field|{let mut sum=0.;for(kk,wk)in [(k,1.-wz),(k+1,wz)]{for(jj,wj)in [(j,1.-wy),(j+1,wy)]{for(ii,wi)in [(a,1.-wx),(b,wx)]{let v=self.fields[field][(kk*self.lat.len()+jj)*self.lon.len()+ii];sum+=wk*wj*wi*if field==0{v.ln()}else{v};}}}if field==0{sum.exp()}else{sum}})}
 /// Diagnostic control: preserve ERA5 below `start`, smoothly remove the
 /// inherited horizontal pressure multiplier, and recover the prescribed
 /// uniform hydrostatic atmosphere above `end`. The transition is an adjustment
 /// region, not a claim of complete geostrophic balance of ERA5 weather.
 pub fn initialize_uniform_upper(&self,m:&Model,d:&Domain,start:f64,end:f64)->Vec<Q>{
  assert!(start>=30000. && end>start);
  let mut u=self.initialize(m,d);let nz=m.background.len();
  for &i in &d.needed{for k in 0..nz{let z=m.vertical.centers[k];if z<=start{continue;}let a=m.background[k];let x=((z-start)/(end-start)).clamp(0.,1.);let weight=1.-x*x*(3.-2.*x);let n=d.index(i,k,nz);let q=u[n];let(pressure,_)=thermo(q,a);let temp=temperature(q,a);let t0=a.p/(a.rho*a.r);let p=a.p*(weight*(pressure/a.p).ln()).exp();let t=t0+weight*(temp-t0);u[n]=conserved([p/(a.r*t*a.rho),weight*q[1]/q[0],weight*q[2]/q[0],weight*q[3]/q[0],p/a.p,0.,0.],a);}}
  u
 }
 /// Hydrostatic version of the uniform-upper control. Above 50 km the legacy
 /// initializer has p/p0 = r, constant with height in each column. If
 /// p=p0 exp(w ln r), dp/dz=-rho0*g*exp(w ln r)+p*w'*ln r.
 /// Therefore rho=-dp/dz/g, rather than retaining T0 through the join.
 /// This removes the imposed continuous vertical force; it is not a claim
 /// that ERA5's lower winds or the discrete fluxes are fully balanced.
 pub fn initialize_hydrostatic_upper(&self,m:&Model,d:&Domain,start:f64,end:f64)->Vec<Q>{
  assert!(start>=50000. && end>start);
  let original=self.initialize(m,d);let mut u=self.initialize_uniform_upper(m,d,start,end);let nz=m.background.len();
  for &i in &d.needed{for k in 0..nz{let z=m.vertical.centers[k];if z<=start || z>=end{continue;}let a=m.background[k];let n=d.index(i,k,nz);let log_ratio=(thermo(original[n],a).0/a.p).ln();let x=(z-start)/(end-start);let weight=1.-x*x*(3.-2.*x);let derivative=-6.*x*(1.-x)/(end-start);let ratio=(weight*log_ratio).exp();let density_ratio=ratio*(1.-a.p/(a.rho*a.g)*derivative*log_ratio);assert!(density_ratio.is_finite() && density_ratio>0.,"Nonpositive hydrostatic join density");let q=u[n];u[n]=conserved([density_ratio,q[1]/q[0],q[2]/q[0],q[3]/q[0],ratio,0.,0.],a);}}
  u
 }
 pub fn initialize(&self,m:&Model,d:&Domain)->Vec<Q>{let nz=m.background.len();let mut u=vec![[0.;7];d.needed.len()*nz];for &i in &d.needed {let up=m.mesh.cells[i].center;let latitude=up[2].asin().to_degrees();let longitude=up[1].atan2(up[0]).to_degrees();let lon=longitude.to_radians();let east=[-lon.sin(),lon.cos(),0.];let north=cross(up,east);let ceiling=if m.vertical.centers[nz-1]<=*self.z.last().unwrap(){*self.z.last().unwrap()}else{30000.};let anchor=self.sample(ceiling,latitude,longitude);let(mut prev_z,mut prev_p,mut prev_t)= (ceiling,anchor[0],anchor[1]);let mut ratio=None;let mut prev_r=287.;
 for k in 0..nz {let h=m.vertical.centers[k];let a=m.background[k];let mut p=self.sample(h.min(ceiling),latitude,longitude);if h>ceiling{let blend=((h-ceiling)/20000.).clamp(0.,1.);let temp=(1.-blend)*anchor[1]+blend*a.p/(a.rho*a.r);let pressure=if let Some(r)=ratio{a.p*r}else{prev_p*(-0.5*a.g*(1./(prev_r*prev_t)+1./(a.r*temp))*(h-prev_z)).exp()};if h>=50000. && ratio.is_none(){ratio=Some(pressure/a.p);}p[0]=pressure;p[1]=temp;for x in &mut p[2..]{*x*=1.-blend;}prev_z=h;prev_p=pressure;prev_t=temp;prev_r=a.r;}
 let density=p[0]/(a.r*p[1]);let vel=add(add(mul(east,p[2]),mul(north,p[3])),mul(up,p[4]));u[d.index(i,k,nz)]=conserved([density/a.rho,vel[0],vel[1],vel[2],p[0]/a.p,0.,0.],a);
 }}u}
}
/// Idealized dry-air absorption; integrated solar power is conserved per column.
pub fn solar_rates(m:&Model,d:&Domain,epoch:f64,absorbed_fraction:f64)->Vec<Q>{let nz=m.background.len();let day=epoch/86400.;let year_day=(day-10957.).rem_euclid(365.2422);let declination=23.44f64.to_radians()*(std::f64::consts::TAU*(year_day-79.)/365.2422).sin();let lon=std::f64::consts::PI-std::f64::consts::TAU*day.fract();let sun=[declination.cos()*lon.cos(),declination.cos()*lon.sin(),declination.sin()];let mut out=vec![[0.;7];d.needed.len()*nz];for &i in &d.owned {let mu=dot(m.mesh.cells[i].center,sun).max(0.);let power=1361.*absorbed_fraction*mu*m.mesh.cells[i].area*RE*RE;let norm=(0..nz).map(|k|(-m.vertical.centers[k]/8000.).exp()*m.volume(i,k)).sum::<f64>();for k in 0..nz{out[d.index(i,k,nz)][4]=power*(-m.vertical.centers[k]/8000.).exp()/norm;}}out}
