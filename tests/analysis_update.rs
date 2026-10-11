use global_neutral_euler::{initialization::AnalysisUpdate,era5::Era5,parallel::Domain,solver::Model};
use num_complex::Complex64;
#[test]
fn window_integrates_once_even_with_irregular_steps_and_stops(){
 let a=AnalysisUpdate{increment:vec![],duration:21600.,lower:30000.,upper:50000.};let mut t=0.;let mut sum=0.;let mut i=0;
 while t<30000.{let dt=(0.13+(i%37) as f64*0.071).min(30000.-t);sum+=dt*a.rate(t,dt);t+=dt;i+=1;}
 assert!((sum-1.).abs()<1e-13);assert_eq!(a.rate(21600.,10.),0.);assert_eq!(a.fraction(-100.),0.);
 for z in [50000.,100000.,450000.]{assert_eq!(AnalysisUpdate::height_weight(z,30000.,50000.),0.);}
}
#[test]
fn window_rejects_fast_adjustment_modes_and_retains_slow_modes(){
 let a=AnalysisUpdate{increment:vec![],duration:21600.,lower:30000.,upper:50000.};let response=|period:f64|{let dt=1.;let mut result=Complex64::new(0.,0.);for i in 0..21600{let t=i as f64;result+=Complex64::from_polar(a.rate(t,dt)*dt,-std::f64::consts::TAU*(t+dt/2.)/period);}result.norm()};
 let fast=response(703.);let slow=response(86400.);assert!(fast<0.001);assert!(slow>0.9);eprintln!("IAU fast response {fast}; slow response {slow}");
}
#[test]
fn upper_forcing_is_zero_and_rank_budgets_add(){
 let m=Model::isothermal(1,70,2000.);let e=Era5::load("data/era5_20250306_23utc_t6_filtered.h5").unwrap();let d=Domain::new(&m.mesh,0,1);let a=AnalysisUpdate::new(&m,&d,&e,21600.,30000.,50000.).unwrap();
 for &i in &d.needed{for k in 25..70{assert_eq!(a.increment[d.index(i,k,70)],[0.;7]);}}
 let total=a.integrals(&m,&d);let mut combined=[0.;8];for rank in 0..2{let p=Domain::new(&m.mesh,rank,2);let update=AnalysisUpdate::new(&m,&p,&e,21600.,30000.,50000.).unwrap();let inv=update.integrals(&m,&p);for j in 0..8{combined[j]+=inv[j];}}
 for j in 0..8{assert!((combined[j]-total[j]).abs()<1e-12*total[j].abs().max(1.));}
 assert!(AnalysisUpdate::new(&m,&d,&e,21600.,30000.,60000.).is_err());
}
