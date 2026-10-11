use global_neutral_euler::{parallel::Domain,physics::*,solver::Model,transport::Transport};
use mpi::traits::*;
#[test]
fn thermal_damping_spectrum_cpu(){
 let uni=mpi::initialize().unwrap();let w=uni.world();assert_eq!(w.size(),1);
 let nz=64;let dz=1000.;let mut m=Model::isothermal(2,nz,dz);let a=Atmos{rho:1e-9,p:1e-9*287.*300.,r:287.,cv:717.5,g:0.};m.background.fill(a);let d=Domain::new(&m.mesh,0,1);let mu=1e-4;let kap=mu*(a.cv+a.r)/0.7;let dt=5.;let mut rows=vec![];
 for mode in [1,4,16,32,63]{
  let theta=std::f64::consts::PI*mode as f64/nz as f64;
  let pattern:Vec<f64>=(0..nz).map(|k|(theta*(k as f64+0.5)).cos()).collect();
  let mut u=m.initial(&d,false);for &i in &d.needed{for k in 0..nz{u[d.index(i,k,nz)]=conserved([1.,0.,0.,0.,1.+pattern[k]/300.,0.,0.],a);}}
  let amplitude=|u:&[Q]|{let(mut num,mut den)=(0.,0.);for &i in &d.owned{for k in 0..nz{let vol=m.volume(i,k);num+=vol*(temperature(u[d.index(i,k,nz)],a)-300.)*pattern[k];den+=vol*pattern[k]*pattern[k];}}num/den};
  let before=amplitude(&u);let mut tr=Transport::new(&m,&d,vec![mu;nz],vec![kap;nz],vec![300.;nz],dt).unwrap();for _ in 0..20{tr.advance(&m,&d,&w,&mut u,dt).unwrap();}
  let ratio=amplitude(&u)/before;let expected=(1.+dt*kap/(a.rho*a.cv)*4.*(theta/2.).sin().powi(2)/(dz*dz)).powi(-20);
  eprintln!("thermal mode={mode} measured={ratio} face_reference={expected}");assert!((ratio-expected).abs()<0.025,"Spectral damping error");rows.extend([mode as f64,ratio,expected]);
 }
 let f=hdf5::File::create("data/transport_face_spectral_validation.h5").unwrap();f.new_dataset_builder().with_data(&rows).create("mode_measured_expected").unwrap();f.new_attr::<f64>().create("passed").unwrap().write_scalar(&1.).unwrap();
}
