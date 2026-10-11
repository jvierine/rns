"""Audit closed IAU runs and serial/MPI regression; no live-file reads."""
from pathlib import Path
import argparse,hashlib
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--serial',type=Path);p.add_argument('--cadence',type=float,default=600);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert(a.run/'run.closed').is_file()
files=sorted(a.run.glob('rank_*.h5'));rows=[];mpi_error=0.
with h5py.File(files[0]) as f:
 assert f.attrs['run_complete']==1
 end=f.attrs['end_time_s'];window=f.attrs['era5_iau_window_s'];assert window>0 and f.attrs['era5_iau_upper_m']<=50000
 keys=sorted(f['frames']);nz=int(f.attrs['layers']);z=f['altitude_m'][:];cv=f['background_cv_j_kg_k'][:];R=f['background_gas_constant_j_kg_k'][:];p0=f['background_pressure_pa'][:];full=f['iau_full_increment_volume_integrals'][:]
 times=np.array([f['frames'][k].attrs['time_s'] for k in keys]);np.testing.assert_allclose(times,np.r_[np.arange(0,end,a.cadence),end],atol=1e-8)
 for key,t in zip(keys,times):
  x=np.clip(t/window,0,1);frac=x-np.sin(2*np.pi*x)/(2*np.pi) if 0<x<1 else x
  pieces=[];ids=[]
  for path in files:
   with h5py.File(path) as g:
    assert g.attrs['run_complete']==1;np.testing.assert_allclose(g['iau_full_increment_volume_integrals'][:],full,rtol=1e-12,atol=1e-9)
    fr=g['frames'][key];assert abs(fr.attrs['iau_applied_fraction']-frac)<1e-13
    np.testing.assert_allclose(fr['iau_applied_volume_integrals'][:],full*frac,rtol=1e-13,atol=1e-9)
    pieces.append(fr['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7));ids.append(g['global_column_id'][:].astype(int))
  q=np.concatenate(pieces)[np.argsort(np.concatenate(ids))];assert np.isfinite(q).all() and (q[:,:,0]>0).all();v=q[:,:,1:4]/q[:,:,0,None];internal=q[:,:,4]-.5*q[:,:,0]*(v*v).sum(-1);assert(internal>0).all();T=internal/q[:,:,0]/cv;assert T.max()<5000
  if t==0:assert abs(v).max()==0
  dp=100*(q[:,:,0]*R*T/p0-1);rows.append([t,frac,T.max(),np.linalg.norm(v,axis=-1)[:,z>=120000].max(),abs(dp[:,z>=120000]).max()])
  if a.serial:
   assert(a.serial/'run.closed').is_file()
   with h5py.File(a.serial/'rank_0000.h5') as s:
    sq=s['frames'][key]['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(q.shape)
    for j in range(7):
     error=np.linalg.norm(q[:,:,j]-sq[:,:,j])/max(np.linalg.norm(sq[:,:,j]),1e-30);mpi_error=max(mpi_error,error)
    assert mpi_error<1e-10
with h5py.File(a.output,'w') as f:
 f.attrs.update(passed=1,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),maximum_serial_mpi_relative_l2=mpi_error,columns='time_s,applied_analysis_fraction,maximum_temperature_K,maximum_upper_total_speed_m_s,maximum_upper_pressure_change_percent')
 f.create_dataset('diagnostics',data=rows);f.create_dataset('full_analysis_volume_integrals',data=full)
 f.attrs['integral_units']='kg,kg m/s,kg m/s,kg m/s,J,kg,kg,J gravitational potential'
print('Passed',len(rows),'frames; serial/MPI relative L2',mpi_error,'maxima',np.array(rows)[:,2:].max(0))
