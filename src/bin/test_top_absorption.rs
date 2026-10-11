//! Isolated outgoing radial acoustic pulse: quantify returned characteristic.
use global_neutral_euler::{parallel::Domain,physics::*,mesh::*,solver::*};
use anyhow::Result;
fn main()->Result<()>{
 let path=std::env::args().nth(1).unwrap_or("data/top_absorption_audit.h5".into());let out=hdf5::File::create(path)?;
 for (name,reflect,sponge) in [("wall",true,false),("wall_sponge",true,true),("open",false,false),("open_sponge",false,true)]{
 let nz=150;let mut m=Model::isothermal(0,nz,2000.);let a=Atmos{rho:1e-5,p:1e-5*287.*300.,r:287.,cv:717.5,g:0.};m.background.fill(a);m.reflect_top=reflect;m.sponge=sponge;m.second_order=true;let c=((1.+a.r/a.cv)*a.p/a.rho).sqrt();let gamma=1.+a.r/a.cv;
 let d=Domain::new(&m.mesh,0,1);let mut u=m.initial(&d,false);for &i in &d.needed{for k in 0..nz{let ep=1e-5*(-0.5*((m.vertical.centers[k]-100000.)/8000.).powi(2)).exp();let v=mul(m.mesh.cells[i].center,c*ep);u[d.index(i,k,nz)]=conserved([1.+ep,v[0],v[1],v[2],1.+gamma*ep,0.,0.],a);}}
 let probe=89;let mut work=RhsWorkspace::new(&m,&d);let mut t=0.;let mut rows=vec![];let(mut incident,mut returned)=(0.,0.);
 while t<1400.{let q=u[d.index(0,probe,nz)];let dp=thermo(q,a).0-a.p;let w=dot([q[1],q[2],q[3]],m.mesh.cells[0].center)/q[0];let plus =0.5*(dp+a.rho*c*w);let minus =0.5*(dp-a.rho*c*w);rows.extend([t,plus,minus]);let dt=m.local_dt(&d,&u).min(1400.-t);if t<450.{incident+=plus*plus*dt;}if t>500.{returned+=minus*minus*dt;}
 m.rhs_cached(&d,&u,&mut work);let mut stage=u.clone();for &i in &d.owned{for k in 0..nz{let n=d.index(i,k,nz);for f in 0..7{stage[n][f]+=dt*work.result()[n][f];}}}m.rhs_cached(&d,&stage,&mut work);for &i in &d.owned{for k in 0..nz{let n=d.index(i,k,nz);for f in 0..7{u[n][f] =0.5*(u[n][f]+stage[n][f]+dt*work.result()[n][f]);}assert!(admissible(u[n],a));}}t+=dt;}
 let g=out.create_group(name)?;g.new_dataset_builder().with_data(&rows).create("time_outgoing_incoming_pa")?;g.new_attr::<f64>().create("reflected_characteristic_energy_ratio")?.write_scalar(&(returned/incident))?;eprintln!("{name}: returned/incident = {}",returned/incident);
 }
 out.new_attr::<u8>().create("complete")?.write_scalar(&1)?;out.flush()?;Ok(())}
