use anyhow::{Result,ensure};
use global_neutral_euler::{parallel::Domain,physics::*,solver::Model,transport::Transport};
fn main()->Result<()>{
 let universe=mpi::initialize().unwrap();let world=universe.world();
 let mut model=Model::isothermal(2,16,2000.);model.background.fill(Atmos{rho:1e-8,p:1e-8*287.*800.,r:287.,cv:717.5,g:9.});let nz=model.background.len();
 let domain=Domain::new(&model.mesh,0,1);let mut initial=model.initial(&domain,false);
 for &i in &domain.owned{for k in 0..nz{let n=domain.index(i,k,nz);let c=model.mesh.cells[i].center;let vel=[10.*c[0]+2.*k as f64,-4.*c[1],3.*(k as f64*0.7).sin()];let a=model.background[k];initial[n]=conserved([1.,vel[0],vel[1],vel[2],1.+0.05*(k as f64*0.3).sin()*c[2],0.001,0.0036],a);}}
 let mut rows=vec![];
 for dt in [0.25,2.,5.]{
  let make=||Transport::new(&model,&domain,vec![1e-3;nz],vec![1.;nz],vec![800.;nz],dt);
  let mut cpu=make()?;let mut gpu=make()?;gpu.enable_metal(&model,&domain)?;
  let mut a=initial.clone();let mut b=initial.clone();cpu.advance(&model,&domain,&world,&mut a,dt)?;gpu.advance(&model,&domain,&world,&mut b,dt)?;
  let mut difference=0f64;let mut ledger=[0.;2];
  for &i in &domain.owned{for k in 0..nz{let n=domain.index(i,k,nz);let scale=[1e-8,1e-7,1e-7,1e-7,0.01,1e-8,1e-8];for f in 0..7{difference=difference.max((a[n][f]-b[n][f]).abs()/scale[f]);}ledger[0]+=(b[n][4]-initial[n][4])*model.volume(i,k);ledger[1]+=initial[n][4]*model.volume(i,k);}}
  ensure!(difference<2e-7,"CPU/Metal transport mismatch {difference}");ensure!(gpu.max_residual<1e-9,"True residual");ensure!((ledger[0]/ledger[1]).abs()<1e-10,"Energy conservation");
  println!("dt={dt} scaled difference={difference} true residual={} refinement={}",gpu.max_residual,gpu.metal_refinement_passes);
  rows.extend([dt,difference,gpu.max_residual,gpu.metal_refinement_passes as f64,ledger[0]/ledger[1]]);
 }
 let args=std::env::args().collect::<Vec<_>>();let path=args.get(1).map(String::as_str).unwrap_or("data/metal_transport_operator_validation.h5");let f=hdf5::File::create(path)?;f.new_dataset_builder().with_data(&rows).create("dt_difference_true_residual_refinement_energy")?;f.new_attr::<u8>().create("passed")?.write_scalar(&1)?;Ok(())
}
