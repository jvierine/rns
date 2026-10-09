//! Closed-archive integrity gate; this does not identify atmospheric modes.
use anyhow::{ensure, Result};
use global_neutral_euler::physics::*;
use std::path::PathBuf;
fn main()->Result<()> {
 let args:Vec<_>=std::env::args().collect();let root=PathBuf::from(&args[1]);
 let equilibrium=args.iter().any(|s|s=="--equilibrium");
 ensure!(root.join("run.closed").exists(),"Archive is not closed");
 let mut paths=std::fs::read_dir(&root)?.filter_map(|x|x.ok().map(|e|e.path())).filter(|p|p.file_name().unwrap().to_string_lossy().starts_with("rank_")).collect::<Vec<_>>();paths.sort();
 let mut all_ids=Vec::new();let mut chronology=None;let mut minimum_density=f64::INFINITY;let mut minimum_pressure=f64::INFINITY;
 for path in &paths {
  let f=hdf5::File::open(path)?;
  ensure!(f.attr("run_complete")?.read_scalar::<f64>()?==1. && f.attr("mpi_size")?.read_scalar::<f64>()?==paths.len() as f64,"Incomplete rank");
  let ids=f.dataset("global_column_id")?.read_raw::<f64>()?;all_ids.extend(ids.iter().map(|&x|x as usize));
  let nz=f.attr("layers")?.read_scalar::<f64>()? as usize;
  let read=|s|->Result<Vec<f64>> {Ok(f.dataset(s)?.read_raw::<f64>()?)};
  let rho=read("background_density_kg_m3")?;let p=read("background_pressure_pa")?;let r=read("background_gas_constant_j_kg_k")?;let cv=read("background_cv_j_kg_k")?;
  let group=f.group("frames")?;let mut keys=group.member_names()?;keys.sort();let mut times=Vec::new();
  for key in keys {
   let g=group.group(&key)?;times.push(g.attr("time_s")?.read_scalar::<f64>()?);
   let state=g.dataset("conserved_density_ecef_momentum_energy_ch4_o2")?.read_raw::<f64>()?;
   ensure!(state.len()==ids.len()*nz*7,"Invalid state shape");
   for (n,q) in state.chunks_exact(7).enumerate() {
    let k=n%nz;let a=Atmos{rho:rho[k],p:p[k],r:r[k],cv:cv[k],g:9.81};let q:Q=q.try_into().unwrap();
    ensure!(admissible(q,a),"Invalid saved state");if equilibrium {ensure!(q==base(a),"Equilibrium drift");}
    minimum_density=minimum_density.min(q[0]);minimum_pressure=minimum_pressure.min(thermo(q,a).0);
   }
   for name in ["mesosphere_80_100km_horizontal_velocity_ecef_m_s","mesosphere_80_100km_radial_velocity_m_s"] {ensure!(g.dataset(name)?.read_raw::<f64>()?.iter().all(|v|v.is_finite()),"Invalid mesospheric diagnostic");}
  }
  ensure!(!times.is_empty() && times[0]==0. && times.windows(2).all(|v|v[1]>v[0]) && *times.last().unwrap()==f.attr("end_time_s")?.read_scalar::<f64>()?,"Invalid chronology");
  if let Some(ref t)=chronology {ensure!(t==&times,"Rank chronology mismatch");} else {chronology=Some(times);}
 }
 all_ids.sort();let mesh=hdf5::File::open(root.join("mesh.h5"))?;let n=mesh.dataset("solid_angle_sr")?.size();ensure!(all_ids==(0..n).collect::<Vec<_>>(),"Rank coverage mismatch");
 let out=hdf5::File::create(root.join("integrity_validation.h5"))?;
 for (name,value) in [("passed",1.),("minimum_density_kg_m3",minimum_density),("minimum_pressure_pa",minimum_pressure),("equilibrium_exact",equilibrium as u8 as f64)] {out.new_attr::<f64>().create(name)?.write_scalar(&value)?;}
 out.new_dataset_builder().with_data(&chronology.unwrap()).create("time_s")?;out.flush()?;
 println!("PASS closed archives; finite positive states, rank coverage and chronology; equilibrium={equilibrium}");Ok(())
}
