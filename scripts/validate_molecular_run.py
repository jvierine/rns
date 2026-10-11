"""Validate a CLOSED molecular pilot including coefficients and transport budgets."""
import argparse,hashlib
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--input',default='data/era5_msis_transport_20250306_23utc.h5');p.add_argument('--output',required=True);a=p.parse_args();assert(a.run/'run.closed').is_file()
rows=[]
with h5py.File(a.run/'rank_0000.h5')as f,h5py.File(a.input)as inp:
 assert f.attrs['run_complete']==1 and f.attrs['molecular_transport_enabled']==1
 for name in ['dynamic_viscosity_pa_s','thermal_conductivity_w_m_k','background_temperature_k']:np.testing.assert_array_equal(f['transport/'+name][:],inp['transport/'+name][:])
 np.testing.assert_array_equal(f['altitude_m'][:],inp['altitude_m'][:]);nz=len(f['altitude_m']);R=f['background_gas_constant_j_kg_k'][:];cv=f['background_cv_j_kg_k'][:]
 for key in f['frames']:
  g=f['frames'][key];q=g['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7);assert np.isfinite(q).all() and np.all(q[:,:,0]>0);internal=q[:,:,4]-.5*(q[:,:,1:4]**2).sum(-1)/q[:,:,0];assert np.all(internal>0);assert g.attrs['transport_max_residual']<5e-9
  totalE=g['global_volume_integrals'][4];work=float(g.attrs['viscous_total_energy_residual_j']);cond=float(g.attrs['conduction_total_energy_residual_j']);assert max(abs(work),abs(cond))<1e-8*abs(totalE)
  rows.append([g.attrs['time_s'],g.attrs['viscous_kinetic_loss_j'],work,cond,g.attrs['transport_max_residual'],g.attrs['transport_max_iterations']])
with h5py.File(a.output,'w')as f:
 f.attrs.update(passed=1,columns='time_s,viscous_kinetic_loss_J,viscous_total_energy_residual_J,conduction_total_energy_residual_J,max_solver_residual,max_iterations',source_sha256=hashlib.sha256((a.run/'rank_0000.h5').read_bytes()).hexdigest());f.create_dataset('budgets',data=rows)
print('Passed molecular run:',len(rows),'frames')
