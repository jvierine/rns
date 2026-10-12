use anyhow::{Result,ensure};
use global_neutral_euler::{compute::{Accelerator,Precision},mesh::*,parallel::Domain,physics::*,solver::{Model,RhsWorkspace}};
fn main()->Result<()>{
 let mut metrics=vec![];
 for scenario in 0..4{let mut model=Model::isothermal(2,12,2000.);model.reconstruct_temperature=true;model.discrete_hydrostatic=true;model.conservative_gravity_energy=true;
 let domain=Domain::new(&model.mesh,0,1);let mut q=model.initial(&domain,false);
 if scenario>0{for &i in &domain.owned{for k in 0..12{let n=domain.index(i,k,12);let c=model.mesh.cells[i].center;let amp=if scenario==2{0.2}else{0.001};q[n]=conserved([1.+amp*c[0],10.*amp*c[1],3.*amp*c[2],amp*k as f64,1.+amp*c[2],if scenario==3{0.01}else{0.},if scenario==3{0.036}else{0.}],model.background[k]);}}}
 let mut cpu=RhsWorkspace::new(&model,&domain);let x=model.rhs_cached(&domain,&q,&mut cpu).to_vec();
 let mut gpu=RhsWorkspace::new(&model,&domain);gpu.accelerator=Accelerator::discover("metal",Precision::F32,0)?;ensure!(gpu.accelerator.is_some(),"Metal required");let y=model.rhs_cached(&domain,&q,&mut gpu).to_vec();ensure!(gpu.accelerator.is_some(),"Fallback");
 let mut error=0f64;for f in 0..7{let scale=x.iter().map(|q|q[f].abs()).fold(0f64,f64::max);let difference=x.iter().zip(&y).map(|(a,b)|(a[f]-b[f]).abs()).fold(0f64,f64::max);if scale>0.{error=error.max(difference/scale);}else{ensure!(difference==0.,"Resting field drift");}}
 ensure!(error<3e-5,"Shared-face CPU/Metal error {error}");println!("scenario={scenario} max normalized RHS difference={error}");metrics.extend([scenario as f64,error]);}
 let a=std::env::args().collect::<Vec<_>>();let f=hdf5::File::create(a.get(1).map(String::as_str).unwrap_or("data/metal_shared_faces_validation.h5"))?;f.new_dataset_builder().with_data(&metrics).create("scenario_maximum_normalized_rhs_difference")?;f.new_attr::<u8>().create("passed")?.write_scalar(&1)?;Ok(())
}
