"""Compare CLOSED evolved ERA5 CPU and native Metal states; save HDF5 evidence."""
import argparse,hashlib,contextlib
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('metal',type=Path);p.add_argument('--end',type=float);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
for path in [a.cpu,a.metal]:assert(path/'run.closed').is_file()
cpus=sorted(a.cpu.glob('rank_*.h5'));metals=sorted(a.metal.glob('rank_*.h5'));assert len(cpus)==len(metals) and len(cpus)>0
paths=[*cpus,*metals];errors=[];times=[]
for cpu_path,metal_path in zip(cpus,metals):
 with h5py.File(cpu_path)as cpu,h5py.File(metal_path)as metal:
  assert cpu.attrs['gpu_flux_enabled']==0 and metal.attrs['gpu_flux_enabled']==1 and metal.attrs['gpu_runtime_fallback']==0
  keys=[k for k in cpu['frames'] if a.end is None or cpu['frames'][k].attrs['time_s']<=a.end];mkeys=[k for k in metal['frames'] if a.end is None or metal['frames'][k].attrs['time_s']<=a.end];assert keys==mkeys;assert a.end is None or cpu['frames'][keys[-1]].attrs['time_s']==a.end;nz=int(cpu.attrs['layers']);R=cpu['background_gas_constant_j_kg_k'][:];cv=cpu['background_cv_j_kg_k'][:];p0=cpu['background_pressure_pa'][:]
  for key in keys:
   arrays=[]
   for f in [cpu,metal]:
    g=f['frames'][key];q=g['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7);assert np.isfinite(q).all() and np.all(q[:,:,0]>0)
    u=q[:,:,1:4]/q[:,:,0,None];internal=q[:,:,4]-.5*q[:,:,0]*(u**2).sum(-1);assert np.all(internal>0)
    reference=f["column_reference_pressure_pa"][:].reshape(q.shape[:2]) if "column_reference_pressure_pa" in f else p0[None,:]
    pressure=internal*R/cv;arrays.append(np.concatenate([q[:,:,0,None],u,(pressure-reference)[:,:,None]],axis=-1))
   x,y=arrays;err=np.sqrt(((x-y)**2).sum(axis=(0,1))/np.maximum((x**2).sum(axis=(0,1)),1e-30));assert err.max()<1e-3,err
   errors.append(err);times.append(float(cpu['frames'][key].attrs['time_s']))
a.output.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.output,'w')as f:
 f.attrs['passed']=1;f.attrs['fields']='density, ECEF velocity x/y/z, pressure perturbation';f.attrs['maximum_relative_l2']=np.max(errors)
 for i,path in enumerate(paths):f.attrs[f'source_{i}_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 f.create_dataset('time_s',data=times);f.create_dataset('relative_l2',data=errors)
print('Max relative L2',np.max(errors))
