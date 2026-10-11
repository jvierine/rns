use global_neutral_euler::{mesh::*,parallel::Domain,physics::*,solver::*};
#[test]
fn closed_sphere_conserves_gas_plus_gravitational_energy(){
 let mut m=Model::isothermal(2,16,5000.);m.reflect_top=true;m.discrete_hydrostatic=true;m.conservative_gravity_energy=true;
 let d=Domain::new(&m.mesh,0,1);let mut u=m.initial(&d,false);
 for &i in &d.owned {for k in 0..16{let x=0.15*m.mesh.cells[i].center[0]*(k as f64*0.4).sin();u[d.index(i,k,16)]=conserved([1.+x,20.*x,50.*x,10.*x,1.-x/2.,0.,0.],m.background[k]);}}
 let rhs=m.rhs(&d,&u);let(mut net,mut magnitude)=(0.,0.);
 for &i in &d.owned {for k in 0..16 {let n=d.index(i,k,16);let phi=m.background[0].g*RE*RE*(1./RE-1./(RE+m.vertical.centers[k]));let de=m.volume(i,k)*(rhs[n][4]+phi*rhs[n][0]);net+=de;magnitude+=de.abs();}}
 assert!(net.abs()<1e-12*magnitude,"energy residual {}",net.abs()/magnitude);
}
