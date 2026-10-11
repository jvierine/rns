use global_neutral_euler::physics::*;
#[test]
fn temperature_ignores_bulk_energy_and_handles_propellant_mixture(){
 let a=Atmos{rho:1.,p:86100.,r:287.,cv:717.5,g:9.81};
 for speed in [0.,7600.] {for (ch,ox) in [(0.,0.),(0.02,0.08)] {
 let q=conserved([1.,speed,0.,0.,1.,ch,ox],a);let gas=(q[0]-q[5]-q[6])*a.r+q[5]*(1.380649e-23/(16.043*1.66053906660e-27))+q[6]*(1.380649e-23/(31.998*1.66053906660e-27));
 assert!((temperature(q,a)-a.p/gas).abs()<1e-8);
 }}
}
