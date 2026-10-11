"""Export closed RNS HDF5 states to a portable GIM layer cache."""
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
from matplotlib import colormaps
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
if not (a.run/'run.closed').is_file():raise ValueError('Run must be closed')
a.output.mkdir(parents=True,exist_ok=True)
(255*colormaps['plasma'](np.linspace(0,1,256))[:,:3]).round().astype('uint8').tofile(a.output/'palette.rgb')
fields=[('pressure','Pressure','Pa'),('pressure_perturbation','Pressure relative to reference atmosphere','Pa'),('up','Upward velocity','m/s'),('east','Eastward velocity','m/s'),('north','Northward velocity','m/s')]
for key,_,_ in fields:(a.output/key).mkdir(exist_ok=True)
with h5py.File(a.run/'mesh.h5')as f:
 v=f['vertices_unit_xyz'][:].reshape(-1,3);tri=f['triangle_vertex_indices'][:].astype(int).reshape(-1,3)
 lon=np.degrees(np.arctan2(v[:,1],v[:,0]));lat=np.degrees(np.arcsin(v[:,2]));xy=np.stack([lon,lat],-1)[tri];xy=np.concatenate([xy,xy[:,2:3]],1);xy.astype('<f4').tofile(a.output/'geometry.f32')
 centers=v[tri].sum(1);centers/=np.linalg.norm(centers,axis=1)[:,None];east=np.stack([-centers[:,1],centers[:,0],np.zeros(len(tri))],-1);east/=np.linalg.norm(east,axis=1)[:,None];north=np.cross(centers,east)
 centers.astype('<f4').tofile(a.output/'centers.f32')
files=sorted(a.run.glob('rank_*.h5'));times=[]
with h5py.File(files[0])as f:
 if f.attrs['run_complete']!=1:raise ValueError('Incomplete run')
 keys=sorted(f['frames']);epoch=float(f.attrs.get('reference_unix_s',0));nz=int(f.attrs['layers']);dz=float(f.attrs['dz_m']);R=f['background_gas_constant_j_kg_k'][:];cv=f['background_cv_j_kg_k'][:];p0=f['background_pressure_pa'][:];altitude=f['altitude_m'][:] if 'altitude_m' in f else (np.arange(nz)+.5)*dz;faces=f['altitude_faces_m'][:] if 'altitude_faces_m' in f else np.arange(nz+1)*dz
reference_pressure=np.broadcast_to(p0,(len(tri),nz)).copy()
for path in files:
 with h5py.File(path) as f:
  if "column_reference_pressure_pa" in f:
   ids=f["global_column_id"][:].astype(int);reference_pressure[ids]=f["column_reference_pressure_pa"][:].reshape(len(ids),nz)
reference_pressure.T.astype("<f4").tofile(a.output/"reference_pressure.f32")
low={key:np.full(nz,np.inf)for key,_,_ in fields};high={key:np.full(nz,-np.inf)for key,_,_ in fields}
for n,key in enumerate(keys):
 q=np.full((len(tri),nz,7),np.nan)
 for path in files:
  with h5py.File(path)as f:
   ids=f['global_column_id'][:].astype(int);g=f['frames'][key];q[ids]=g['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(len(ids),nz,7);t=float(g.attrs['time_s'])
 if not np.isfinite(q).all()or np.any(q[:,:,0]<=0):raise ValueError('Invalid states')
 velocity=q[:,:,1:4]/q[:,:,0,None];ambient=q[:,:,0]-q[:,:,5]-q[:,:,6];rch=1.380649e-23/(16.043*1.66053906660e-27);ro=1.380649e-23/(31.998*1.66053906660e-27)
 gas=ambient*R+q[:,:,5]*rch+q[:,:,6]*ro;heatcap=ambient*cv+q[:,:,5]*3*rch+q[:,:,6]*2.5*ro
 pressure=(q[:,:,4]-.5*q[:,:,0]*(velocity**2).sum(-1))*gas/heatcap
 values=dict(pressure=pressure,pressure_perturbation=pressure-reference_pressure,up=(velocity*centers[:,None,:]).sum(-1),east=(velocity*east[:,None,:]).sum(-1),north=(velocity*north[:,None,:]).sum(-1))
 # Verify equation-of-state extraction against the solver's independent saved slice.
 with h5py.File(files[0])as f:
  ids=f['global_column_id'][:].astype(int);np.testing.assert_allclose(values['pressure_perturbation'][ids,0],f['frames'][key]['surface_pressure_perturbation_pa'][:],atol=1e-8,rtol=1e-10)
 for field,_,_ in fields:
  x=values[field].T;assert np.isfinite(x).all();low[field]=np.minimum(low[field],x.min(1));high[field]=np.maximum(high[field],x.max(1));x.astype('<f4').tofile(a.output/field/f'{n:04d}.f32')
 times.append(t)
metadata=[]
for key,label,units in fields:
 if key=='pressure':limits=np.stack([low[key],high[key]],-1);limits[:,1]=np.maximum(limits[:,1],limits[:,0]+1e-6);cut=[float(low[key].min()),float(high[key].max())]
 else:
  peak=np.maximum(np.maximum(abs(low[key]),abs(high[key])),1e-8);limits=np.stack([-peak,peak],-1);cut=[-float(peak.max()),float(peak.max())]
 metadata.append(dict(key=key,label=label,units=units,limits=limits.tolist(),cut_limits=cut))
meta=dict(schema='rns.layer/v2',fields=metadata,altitude_m=altitude.tolist(),altitude_faces_m=faces.tolist(),times_s=times,reference_unix_s=epoch,palette='plasma',camera=dict(yaw=0,pitch=0),cell_count=len(tri),layout='altitude, cell; little-endian float32',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [a.run/'mesh.h5',*files]})
(a.output/'metadata.json').write_text(json.dumps(meta,indent=2));print(f'{len(times)} frames, {len(tri)} cells, {nz} levels, five fields')
