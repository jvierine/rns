"""Nominal molecular transport on the frozen ERA5–MSIS reference profile."""
import argparse,hashlib,shutil
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',default='data/era5_msis_20250306_23utc.h5');p.add_argument('--output',default='data/era5_msis_transport_20250306_23utc.h5');a=p.parse_args();assert not Path(a.output).exists();shutil.copyfile(a.input,a.output)
with h5py.File(a.output,'r+')as f:
 rho=f['density_kg_m3'][:];R=f['gas_constant_j_kg_k'][:];cv=f['cv_j_kg_k'][:];T=f['pressure_pa'][:]/(rho*R);cp=cv+R;mu=3.563e-7*T**.69;kappa=mu*cp/.7;nu=mu/rho;chi=kappa/(rho*cp)
 tr=f.create_group('transport')
 for name,x in [('background_temperature_k',T),('dynamic_viscosity_pa_s',mu),('thermal_conductivity_w_m_k',kappa),('kinematic_viscosity_m2_s',nu),('thermal_diffusivity_m2_s',chi)]:assert np.isfinite(x).all() and np.all(x>0);tr.create_dataset(name,data=x)
 tr.attrs.update(viscosity_reference='Fomichev et al. 2002 doi:10.1029/2001JD000479; mu=3.563e-7 T0^0.69 Pa s',Pr=.7,conduction='kappa=mu*(cv+R)/Pr; Fourier flux of T-T0, maintaining prescribed background',assumptions='Frozen ambient coefficients; nominal Prandtl closure, not species-resolved collision integrals; zero bulk viscosity; continuum constitutive law becomes unreliable for wavelength-scale Knudsen number near unity',parent_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 # Local uniform-medium acoustic estimate only, not a gravity-wave dispersion relation.
 lam=np.array([50000.,100000.,200000.,500000.]);gamma=1+R/cv;rate=.5*((4/3)*nu[:,None]+(gamma[:,None]-1)*chi[:,None])*(2*np.pi/lam[None,:])**2
 tr.create_dataset('acoustic_wavelength_m',data=lam);tr.create_dataset('local_acoustic_amplitude_efolding_s',data=1/rate)
 for height in [100,150,200,300]:
  k=np.argmin(abs(f['altitude_m'][:]/1000-height));print(f['altitude_m'][k]/1000,'km: nu=',nu[k],'m2/s; chi=',chi[k],'m2/s; 100km acoustic amplitude e-fold=',1/rate[k,1],'s')
print(a.output)
