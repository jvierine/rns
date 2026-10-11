use global_neutral_euler::{parallel::Domain,physics::*,solver::Model,transport::Transport};
use mpi::{collective::SystemOperation,traits::*};
fn totals(m:&Model,d:&Domain,w:&impl CommunicatorCollectives,u:&[Q])->[f64;6]{
 let mut local=[0.;6];let nz=m.background.len();for &i in &d.owned{for k in 0..nz{let q=u[d.index(i,k,nz)];let v=m.volume(i,k);for f in 0..5{local[f]+=q[f]*v;}local[5]+=0.5*(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0]*v;}}
 let mut total=[0.;6];w.all_reduce_into(&local,&mut total,SystemOperation::sum());total
}
#[test]
fn cpu_molecular_transport_budgets_and_damping(){
 let universe=mpi::initialize().unwrap();let world=universe.world();let mut m=Model::isothermal(1,64,1000.);
 let a=Atmos{rho:1e-9,p:1e-9*287.*300.,r:287.,cv:717.5,g:9.81};m.background.fill(a);
 let d=Domain::new(&m.mesh,world.rank()as usize,world.size()as usize);let nz=m.background.len();
 let mu=1e-4;let kap=mu*(a.cv+a.r)/0.7;
 let make=||Transport::new(&m,&d,vec![mu;nz],vec![kap;nz],vec![300.;nz],5.).unwrap();
 let mut u=d.needed.iter().flat_map(|_|(0..nz).map(|_|conserved([1.,5.,-2.,3.,1.,0.,0.],a))).collect::<Vec<_>>();let initial=u.clone();let mut tr=make();tr.advance(&m,&d,&world,&mut u,5.).unwrap();
 assert!(u.iter().zip(&initial).all(|(x,y)|x.iter().zip(y).all(|(a,b)|(a-b).abs()<1e-12*b.abs().max(1e-30))),"Uniform translation must be unchanged");
 for &i in &d.needed{for k in 0..nz{let v=30.*(std::f64::consts::PI*m.vertical.centers[k]/64000.).cos();u[d.index(i,k,nz)]=conserved([1.,v,0.,0.,1.,0.,0.],a);}}
 let before=totals(&m,&d,&world,&u);let mut tr=make();for _ in 0..20{tr.advance(&m,&d,&world,&mut u,5.).unwrap();}let after=totals(&m,&d,&world,&u);
 assert!(after[5]<before[5]*0.98,"Velocity mode must damp");assert!((after[0]/before[0]-1.).abs()<1e-13);assert!((after[4]/before[4]-1.).abs()<1e-12,"Viscosity/conduction must conserve total energy");
 let scale=before[0]*30.;for f in 1..4{assert!((after[f]-before[f]).abs()<1e-10*scale,"Momentum conservation");}
 let heat=(after[4]-after[5])-(before[4]-before[5]);assert!((heat/(before[5]-after[5])-1.).abs()<1e-8,"Lost kinetic energy must become heat");
 // Long vertical cosine mode, local continuum thermal diffusion prediction.
 let wave=std::f64::consts::PI/64000.;
 let moment=|u:&[Q]|{let mut v=[0.;2];for &i in &d.owned{for k in 0..nz{let q=u[d.index(i,k,nz)];let temp=thermo(q,a).0/(q[0]*a.r);let c=(wave*m.vertical.centers[k]).cos();v[0]+=(temp-300.)*c*m.volume(i,k);v[1]+=c*c*m.volume(i,k);}}let mut t=[0.;2];world.all_reduce_into(&v,&mut t,SystemOperation::sum());t[0]/t[1]};
 for &i in &d.needed{for k in 0..nz{let temp=300.+(wave*m.vertical.centers[k]).cos();u[d.index(i,k,nz)]=conserved([1.,0.,0.,0.,temp/300.,0.,0.],a);}}
 let mut thermal=make();let start=moment(&u);for _ in 0..20{thermal.advance(&m,&d,&world,&mut u,5.).unwrap();}let ratio=moment(&u)/start;let expected=(-kap/(a.rho*a.cv)*wave*wave*100.).exp();assert!((ratio-expected).abs()<0.01,"Thermal cosine damping: {ratio} vs {expected}");

 // Coupled acoustic attenuation: matched inviscid and viscous/conductive runs.
 let mut acoustic=Model::isothermal(1,128,500.);let aa=Atmos{g:0.,..a};acoustic.background.fill(aa);acoustic.reflect_top=true;
 let ad=Domain::new(&acoustic.mesh,world.rank()as usize,world.size()as usize);let an=acoustic.background.len();let ak=2.*std::f64::consts::PI/64000.;let gamma=1.+aa.r/aa.cv;
 let mut initial=acoustic.initial(&ad,false);for &i in &ad.needed{for k in 0..an{let amp=1e-5*(ak*acoustic.vertical.centers[k]).cos();initial[ad.index(i,k,an)]=conserved([1.+amp/gamma,0.,0.,0.,1.+amp,0.,0.],aa);}}
 let mut energies:Vec<f64>=vec![];
 for enabled in [false,true] {
  let mut state=initial.clone();let mut visc=Transport::new(&acoustic,&ad,vec![mu;an],vec![kap;an],vec![300.;an],0.5).unwrap();
  let mut work=global_neutral_euler::solver::RhsWorkspace::new(&acoustic,&ad);let mut exchange=ad.exchange_workspace(an);
  for step in 0..2000 {
   ad.exchange_cached(&world,&mut state,an,&mut exchange);let rhs=acoustic.rhs_cached(&ad,&state,&mut work);let mut stage=state.clone();for &i in &ad.owned{for k in 0..an{let n=ad.index(i,k,an);for f in 0..7{stage[n][f]+=0.1*rhs[n][f];}}}
   ad.exchange_cached(&world,&mut stage,an,&mut exchange);let rhs=acoustic.rhs_cached(&ad,&stage,&mut work);for &i in &ad.owned{for k in 0..an{let n=ad.index(i,k,an);for f in 0..7{state[n][f]=0.5*(state[n][f]+stage[n][f]+0.1*rhs[n][f]);}assert!(admissible(state[n],aa));}}
   if enabled && (step+1)%5==0 {visc.advance(&acoustic,&ad,&world,&mut state,0.5).unwrap();}
  }
  let mut energy=0.;for &i in &ad.owned{for k in 0..an{let q=state[ad.index(i,k,an)];let pp=thermo(q,aa).0-aa.p;energy+=acoustic.volume(i,k)*(pp*pp/(gamma*aa.p)+(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0]);}}
  let mut total=0.;world.all_reduce_into(&energy,&mut total,SystemOperation::sum());energies.push(total);
 }
 let acoustic_ratio=(energies[1]/energies[0]).sqrt();let chi=kap/(aa.rho*(aa.cv+aa.r));let acoustic_prediction=(-0.5*((4./3.)*mu/aa.rho+(gamma-1.)*chi)*ak*ak*200.).exp();
 assert!((acoustic_ratio/acoustic_prediction-1.).abs()<0.05,"Acoustic attenuation: {acoustic_ratio} vs {acoustic_prediction}");
 if world.rank()==0{eprintln!("Acoustic amplitude ratio {acoustic_ratio}, predicted {acoustic_prediction}");}
 if world.rank()==0{eprintln!("Kinetic-energy ratio {}; thermal amplitude ratio {ratio}, predicted {expected}; max solver residual {}",after[5]/before[5],thermal.max_residual);if let Ok(path)=std::env::var("RNS_TRANSPORT_EVIDENCE"){let f=hdf5::File::create(path).unwrap();for(n,x)in[("passed",1.),("kinetic_energy_ratio",after[5]/before[5]),("thermal_amplitude_ratio",ratio),("thermal_predicted_ratio",expected),("total_energy_relative_error",(after[4]/before[4]-1.).abs()),("maximum_residual",thermal.max_residual),("acoustic_amplitude_ratio",acoustic_ratio),("acoustic_predicted_ratio",acoustic_prediction)]{f.new_attr::<f64>().create(n).unwrap().write_scalar(&x).unwrap();}}}
}
