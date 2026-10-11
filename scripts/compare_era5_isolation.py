"""Compare CLOSED quiet, ERA5 and solar isolation controls."""
from pathlib import Path
import argparse,hashlib
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--filtered-run',type=Path);p.add_argument('--iau-run',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cases=[('quiet','Quiet reference'),('era5_thermodynamics','ERA5 temperature/pressure'),('era5_winds','ERA5 + winds'),('solar_only','Solar-only control')]
if a.filtered_run:cases.append((str(a.filtered_run.resolve()),'Filtered ERA5 + winds'))
if a.iau_run:cases.append((str(a.iau_run.resolve()),'Gradual ERA5 IAU'))
results=[]
for name,label in cases:
 run=a.campaign/name;assert(run/'run.closed').is_file()
 with h5py.File(run/'mesh.h5') as f:
  v=f['vertices_unit_xyz'][:].reshape(-1,3);tri=f['triangle_vertex_indices'][:].reshape(-1,3).astype(int);up=v[tri].sum(1);up/=np.linalg.norm(up,axis=1)[:,None]
 with h5py.File(run/'rank_0000.h5') as f:
  if name=='solar_only':label='ERA5 + solar (no winds)' if 'column_reference_pressure_pa' in f else 'Uniform reference + solar'
  assert f.attrs['run_complete']==1;nz=int(f.attrs['layers']);ids=f['global_column_id'][:].astype(int);z=f['altitude_m'][:];R=f['background_gas_constant_j_kg_k'][:];cv=f['background_cv_j_kg_k'][:]
  pref=f['column_reference_pressure_pa'][:].reshape(-1,nz) if 'column_reference_pressure_pa'in f else f['background_pressure_pa'][:][None,:]
  rows=[]
  for key in sorted(f['frames']):
   fr=f['frames'][key];q=fr['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7);assert np.isfinite(q).all()and(q[:,:,0]>0).all();vel=q[:,:,1:4]/q[:,:,0][:,:,None];T=(q[:,:,4]-.5*q[:,:,0]*(vel**2).sum(-1))/(q[:,:,0]*cv);assert (T>0).all();w=np.einsum('ijk,ik->ij',vel,up[ids]);pressure=q[:,:,0]*R*T;dp=100*(pressure/pref-1)
   rows.append([fr.attrs['time_s'],abs(w).max(),abs(w[:,z>=120000]).max(),T.max(),abs(dp[:,z<30000]).max(),abs(dp[:,z>=120000]).max()])
  results.append((name,label,np.array(rows)))
with h5py.File(a.output/'comparison.h5','w')as f:
 f.attrs.update(columns='time_s,max_abs_w_m_s,max_abs_w_above120km_m_s,max_T_K,max_abs_pressure_percent_below30km,max_abs_pressure_percent_above120km',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 for name,label,r in results:
  d=f.create_dataset(Path(name).name,data=r);d.attrs['label']=label;d.attrs['source_run']=str((a.campaign/name).resolve());d.attrs['source_sha256']=hashlib.sha256((a.campaign/name/'rank_0000.h5').read_bytes()).hexdigest()
plt.rcParams.update({'font.size':10,'figure.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(10,6.5),sharex=True,layout='constrained')
for name,label,r in results:
 for ax,j in zip(axs.flat,[2,3,4,5]):ax.plot(r[:,0]/3600,r[:,j],label=label,lw=1.5)
for ax,title in zip(axs.flat,['Vertical speed above 120 km [m/s]','Maximum temperature [K]','Pressure change below 30 km [%]','Pressure change above 120 km [%]']):ax.set_title(title);ax.grid(alpha=.2);ax.set_xlim(0,max(r[-1,0] for _,_,r in results)/3600)
for ax in axs[1]:ax.set_xlabel('Elapsed time [h]')
axs[0,0].legend(fontsize=8);fig.suptitle('Initial-state isolation · same solver and grid',fontsize=12);fig.savefig(a.output/'comparison.png',dpi=160);fig.savefig(a.output/'comparison.pdf');plt.close(fig)
for name,label,r in results:print(name,'all-time maxima:',r[:,1:].max(0).tolist())
