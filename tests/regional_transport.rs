use global_neutral_euler::{parallel::Domain,solver::Model,transport::Transport};
#[test]
fn regional_resting_transport_handles_zero_residual_stencils(){
 let universe=mpi::initialize().unwrap();let world=universe.world();
 let m=Model::isothermal(1,8,1000.);let nz=m.background.len();
 let d=Domain::selected(&m.mesh,0,1,&[0]);
 let mut q=m.initial(&d,false);let before=q.clone();
 let mut tr=Transport::new(&m,&d,vec![1e-5;nz],vec![0.01;nz],vec![300.;nz],1.).unwrap();
 tr.advance(&m,&d,&world,&mut q,1.).unwrap();
 for (a,b)in q.iter().zip(before){for k in 0..7{assert!((a[k]-b[k]).abs()<=1e-12*b[k].abs().max(1e-30));}}
}
