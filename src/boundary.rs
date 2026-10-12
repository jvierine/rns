//! Configurable absorbing layers for a rectangular spherical region.
use crate::mesh::{V,RE};
use anyhow::{Result,ensure};
#[derive(Clone,Copy,Debug)]
pub struct AbsorbingLayers {
 pub bounds_deg:[f64;4],
 pub lateral_width_m:f64,
 pub lateral_timescale_s:f64,
 pub top_start_m:f64,
 pub top_end_m:f64,
 pub top_timescale_s:f64,
 pub lateral_enabled:bool,
 pub top_enabled:bool,
}
impl AbsorbingLayers {
 pub fn validate(&self)->Result<()> {
  let [s,n,w,e]=self.bounds_deg;
  ensure!([s,n,w,e,self.lateral_width_m,self.lateral_timescale_s,self.top_start_m,self.top_end_m,self.top_timescale_s].iter().all(|x|x.is_finite()),"Nonfinite boundary configuration");
  ensure!(s>=-90. && n<=90. && s<n && w>=-180. && e<=180. && w<e,"Invalid regional bounds (dateline crossing requires a separate geometry)");
  ensure!(self.lateral_width_m>0. && self.lateral_timescale_s>0. && self.top_start_m>=0. && self.top_end_m>self.top_start_m && self.top_timescale_s>0.,"Invalid absorbing layer widths/timescales");Ok(())
 }
 pub fn lateral_rate(&self,p:V)->f64 {
  if !self.lateral_enabled{return 0.;}
  let lat=p[2].clamp(-1.,1.).asin();let lon=p[1].atan2(p[0]);let b=self.bounds_deg.map(f64::to_radians);
  let distance=RE*((lat-b[0]).min(b[1]-lat)).min(((lon-b[2]).min(b[3]-lon))*lat.cos());
  ((self.lateral_width_m-distance)/self.lateral_width_m).clamp(0.,1.).powi(2)/self.lateral_timescale_s
 }
 pub fn top_rate(&self,z:f64)->f64 {
  if !self.top_enabled{return 0.;}
  ((z-self.top_start_m)/(self.top_end_m-self.top_start_m)).clamp(0.,1.).powi(2)/self.top_timescale_s
 }
}
#[cfg(test)]
mod tests {
 use super::*;
 fn config()->AbsorbingLayers {AbsorbingLayers{bounds_deg:[-35.,55.,-115.,-15.],lateral_width_m:400000.,lateral_timescale_s:100.,top_start_m:650000.,top_end_m:1000000.,top_timescale_s:60.,lateral_enabled:true,top_enabled:true}}
 fn point(lat:f64,lon:f64)->V{let (lat,lon)=(lat.to_radians(),lon.to_radians());[lat.cos()*lon.cos(),lat.cos()*lon.sin(),lat.sin()]}
 #[test]fn rates_are_zero_in_interior_and_smooth_at_entry(){let c=config();c.validate().unwrap();assert_eq!(c.lateral_rate(point(0.,-60.)),0.);assert_eq!(c.top_rate(650000.),0.);assert!((c.top_rate(1000000.)-1./60.).abs()<1e-15);assert!((c.lateral_rate(point(-35.,-60.))-0.01).abs()<1e-14);let h=1.;assert!(c.top_rate(650000.+h)<1e-12);}
 #[test]fn can_disable_layers_and_reject_invalid_geometry(){let mut c=config();c.top_enabled=false;c.lateral_enabled=false;assert_eq!(c.top_rate(1000000.),0.);assert_eq!(c.lateral_rate(point(-35.,-60.)),0.);c.bounds_deg[0]=60.;assert!(c.validate().is_err());}
}
