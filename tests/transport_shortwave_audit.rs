use global_neutral_euler::{parallel::Domain,physics::*,solver::Model,transport::Transport};
use mpi::traits::*;
#[test]
fn audit_alternating_vertical_velocity_damping(){
 let universe=mpi::initialize().unwrap();let world=universe.world();assert_eq!(world.size(),1);
 let nz=64;let dz=1000.;let mut m=Model::isothermal(0,nz,dz);
 let a=Atmos{rho:1e-9,p:1e-9*287.*300.,r:287.,cv:717.5,g:0.};m.background.fill(a);
 let d=Domain::new(&m.mesh,0,1);let mu=1e-4;let kap=mu*(a.cv+a.r)/0.7;
 let mut tr=Transport::new(&m,&d,vec![mu;nz],vec![kap;nz],vec![300.;nz],5.).unwrap();
 let mut u=m.initial(&d,false);for &i in &d.needed{for k in 0..nz{u[d.index(i,k,nz)]=conserved([1.,if k%2==0{1.}else{-1.},0.,0.,1.,0.,0.],a);}}
 let amplitude=|u:&[Q]|{let(mut top,mut bot)=(0.,0.);for &i in &d.owned{for k in 0..nz{let n=d.index(i,k,nz);let sign=if k%2==0{1.}else{-1.};let vol=m.volume(i,k);top+=vol*u[n][1]/u[n][0]*sign;bot+=vol;}}top/bot};
 let before=amplitude(&u);for _ in 0..20{tr.advance(&m,&d,&world,&mut u,5.).unwrap();}let ratio=amplitude(&u)/before;
 let face_reference=(1.+5.*4.*mu/a.rho/(dz*dz)).powi(-20);
 eprintln!("ALTERNATING_MODE amplitude_ratio={ratio} nearest_face_reference={face_reference}");
 let path="data/transport_shortwave_corrected_validation.h5";let f=hdf5::File::create(path).unwrap();for(name,value)in[("duration_s",100.),("dz_m",dz),("kinematic_viscosity_m2_s",mu/a.rho),("amplitude_ratio",ratio),("nearest_face_backward_euler_reference",face_reference)]{f.new_attr::<f64>().create(name).unwrap().write_scalar(&value).unwrap();}
 assert!(ratio.abs()<0.01,"Grid-scale pattern must damp: {ratio}");
}
