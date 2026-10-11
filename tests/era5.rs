use global_neutral_euler::{era5::{Era5,solar_rates},mesh::{dot,RE},parallel::Domain,physics::thermo,solver::Model};
#[test]
fn era5_weather_grid_retains_upper_level_pressure_and_wind(){
 let e=Era5::load("data/era5_20250306_23utc.h5").unwrap();let m=Model::isothermal(1,20,2000.);let d=Domain::new(&m.mesh,0,1);let u=e.initialize(&m,&d);
 for &i in &d.owned {let p=m.mesh.cells[i].center;let sample=e.sample(39000.,p[2].asin().to_degrees(),p[1].atan2(p[0]).to_degrees());let q=u[d.index(i,19,20)];let pressure=thermo(q,m.background[19]).0;assert!((pressure/sample[0]-1.).abs()<1e-12);let lon=p[1].atan2(p[0]);let east=[-lon.sin(),lon.cos(),0.];let v=[q[1]/q[0],q[2]/q[0],q[3]/q[0]];assert!((dot(v,east)-sample[2]).abs()<1e-10);}
}
#[test]
fn solar_column_budget_and_night_exclusion(){
 let m=Model::isothermal(3,20,2000.);let d=Domain::new(&m.mesh,0,1);let source=solar_rates(&m,&d,1741302000.,0.2);let power=d.owned.iter().map(|&i|(0..20).map(|k|source[d.index(i,k,20)][4]*m.volume(i,k)).sum::<f64>()).sum::<f64>();let expected=std::f64::consts::PI*RE*RE*1361.*0.2;assert!((power/expected-1.).abs()<0.01);assert!(d.owned.iter().any(|&i|(0..20).all(|k|source[d.index(i,k,20)][4]==0.)));assert!(source.iter().all(|q|q[4]>=0.&&q.iter().enumerate().all(|(f,x)|f==4||*x==0.)));
}

#[test]
fn uniform_upper_control_retains_lower_weather_and_restores_upper_background(){
 let e=Era5::load("data/era5_20250306_23utc.h5").unwrap();let m=Model::isothermal(1,30,5000.);let d=Domain::new(&m.mesh,0,1);let original=e.initialize(&m,&d);let u=e.initialize_uniform_upper(&m,&d,50000.,120000.);
 for &i in &d.owned{for k in 0..30{let n=d.index(i,k,30);if m.vertical.centers[k]<=50000.{assert_eq!(u[n],original[n]);}if m.vertical.centers[k]>=120000.{let a=m.background[k];assert!((u[n][0]/a.rho-1.).abs()<1e-14);assert!(u[n][1..4].iter().all(|x|*x==0.));assert!((thermo(u[n],a).0/a.p-1.).abs()<1e-14);}}}
}

#[test]
fn hydrostatic_upper_join_supplies_density_for_pressure_gradient(){
 use global_neutral_euler::{physics::{Atmos,temperature},solver::RhsWorkspace,vertical::VerticalGrid};
 let e=Era5::load("data/era5_20250306_23utc.h5").unwrap();let f=hdf5::File::open("data/era5_msis_transport_450km_control.h5").unwrap();
 let read=|n:&str| f.dataset(n).unwrap().read_raw::<f64>().unwrap();let z=read("altitude_m");let nz=z.len();let mut m=Model::isothermal(2,nz,2000.);m.set_vertical(VerticalGrid::new(read("altitude_faces_m")));
 let rho=read("density_kg_m3");let p=read("pressure_pa");let r=read("gas_constant_j_kg_k");let cv=read("cv_j_kg_k");let g=read("gravity_m_s2");for k in 0..nz{m.background[k]=Atmos{rho:rho[k],p:p[k],r:r[k],cv:cv[k],g:g[k]};}
 m.reconstruct_temperature=true;m.discrete_hydrostatic=true;let d=Domain::new(&m.mesh,0,1);let original=e.initialize(&m,&d);let plain=e.initialize_uniform_upper(&m,&d,50000.,120000.);let corrected=e.initialize_hydrostatic_upper(&m,&d,50000.,120000.);
 let mut residual=0f64;
 for &i in &d.owned{for k in 0..nz{let n=d.index(i,k,nz);let a=m.background[k];if z[k]<=50000. || z[k]>=120000.{assert_eq!(plain[n],corrected[n]);continue;}let x=(z[k]-50000.)/70000.;let w=1.-x*x*(3.-2.*x);let wp=-6.*x*(1.-x)/70000.;let lr=(thermo(original[n],a).0/a.p).ln();let ratio=(w*lr).exp();let dp=-a.rho*a.g*ratio+a.p*ratio*wp*lr;residual=residual.max((dp+corrected[n][0]*a.g).abs()/(a.rho*a.g));assert!(temperature(corrected[n],a)>0.);assert!((thermo(corrected[n],a).0/thermo(plain[n],a).0-1.).abs()<1e-14);}}
 assert!(residual<1e-14);
 let mut work=RhsWorkspace::new(&m,&d);let a=m.rhs_cached(&d,&plain,&mut work).to_vec();let b=m.rhs_cached(&d,&corrected,&mut work).to_vec();let mut peaks=[0f64;2];
 for &i in &d.owned{for k in 0..nz{if z[k]>50000. && z[k]<120000.{let n=d.index(i,k,nz);let up=m.mesh.cells[i].center;for (j,(rhs,u)) in [(&a,&plain),(&b,&corrected)].iter().enumerate(){peaks[j]=peaks[j].max(dot([rhs[n][1],rhs[n][2],rhs[n][3]],up).abs()/u[n][0]);}}}}
 eprintln!("Join radial acceleration plain/corrected {peaks:?} m/s2; continuous balance {residual}");assert!(peaks[1]<0.5*peaks[0]);
 let out=hdf5::File::create("data/era5_hydrostatic_join_validation.h5").unwrap();out.new_dataset_builder().with_data(&peaks).create("initial_radial_acceleration_max_m_s2").unwrap();out.new_attr::<f64>().create("continuous_balance_relative_residual").unwrap().write_scalar(&residual).unwrap();
}
