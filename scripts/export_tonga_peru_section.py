"""Extract a great-circle section from CLOSED Rust/MPI Tonga archives."""
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial import cKDTree
p=argparse.ArgumentParser();p.add_argument('--lamb',required=True);p.add_argument('--mesosphere',required=True);p.add_argument('--output',required=True);a=p.parse_args()
def unit(lat,lon):
 lat,lon=np.deg2rad([lat,lon]);return np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
s=unit(-20.54,-175.38);target=unit(-11.9,-76.8);angle=np.arccos(s@target);direction=(target-np.cos(angle)*s)/np.sin(angle)
distance=np.arange(0.,12000.1,50.);theta=distance/6371.;points=np.cos(theta)[:,None]*s+np.sin(theta)[:,None]*direction;tangents=-np.sin(theta)[:,None]*s+np.cos(theta)[:,None]*direction
with h5py.File(a.output,'w') as out:
 out['distance_km']=distance;out['section_unit_xyz']=points;out.attrs['jicamarca_distance_km']=angle*6371.;out.attrs['source_lat_lon_deg']=[-20.54,-175.38];out.attrs['target_lat_lon_deg']=[-11.9,-76.8];out.attrs['scope']='Actual archived nonlinear Rust/MPI fields; positive 3-nearest-column inverse-distance interpolation; no observational fit or verified mode attribution';out.attrs['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for name,root in [('lamb',Path(a.lamb)),('mesosphere',Path(a.mesosphere))]:
  assert (root/'run.closed').exists(),f'{root} is not closed'
  with h5py.File(root/'mesh.h5') as f:centers=f['center_unit_xyz'][:].reshape(-1,3)
  d,ids=cKDTree(centers).query(points,k=3);weights=1/np.maximum(d,1e-12)**2;weights/=weights.sum(axis=1)[:,None];needed=np.unique(ids)
  g=out.create_group(name);g['interpolation_column_ids']=ids;g['interpolation_weights']=weights;g['nearest_column_distance_km']=2*np.arcsin(np.minimum(d[:,0]/2,1))*6371.;times=None;pressure=None;wind=None;found=set();hashes={}
  for file in sorted(root.glob('rank_*.h5')):
   with h5py.File(file) as f:
    assert f.attrs['run_complete']==1;assert 'display_subset' not in f.attrs,'Need full 3D states'
    global_ids=f['global_column_id'][:].astype(int);local=np.flatnonzero(np.isin(global_ids,needed))
    if not len(local):continue
    keys=sorted(f['frames']);t=np.array([f['frames'][k].attrs['time_s'] for k in keys]);nz=int(f.attrs['layers']);dz=float(f.attrs['dz_m']);p0=f['background_pressure_pa'][:];rho0=f['background_density_kg_m3'][:]
    ratio=f['background_gas_constant_j_kg_k'][:]/f['background_cv_j_kg_k'][:] if 'background_cv_j_kg_k' in f else np.full(nz,.4)
    if times is None:
     times=t;pressure=np.zeros((len(t),len(points),nz));wind=np.zeros_like(pressure);g['altitude_km']=(np.arange(nz)+.5)*dz/1000;g['background_pressure_pa']=p0;g['background_density_kg_m3']=rho0;g.attrs['run_config_json']=json.dumps({k:float(v) for k,v in f.attrs.items()})
    assert np.array_equal(times,t)
    for li in local:found.add(int(global_ids[li]))
    for it,k in enumerate(keys):
     q=f['frames'][k]['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7);assert np.isfinite(q).all() and (q[:,:,0]>0).all()
     for li in local:
      gid=global_ids[li];loc=np.argwhere(ids==gid);state=q[li];vel=state[:,1:4]/state[:,0,None];pr=(state[:,4]-np.sum(state[:,1:4]**2,axis=1)/(2*state[:,0]))*ratio-p0
      for ip,j in loc:
       pressure[it,ip]+=weights[ip,j]*pr;wind[it,ip]+=weights[ip,j]*(vel@tangents[ip])
    # Hash only actually used full files; does not read any live archive.
   hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
  assert found==set(needed.tolist());assert np.isfinite(pressure).all() and np.isfinite(wind).all();assert np.allclose(weights.sum(axis=1),1)
  g['time_s']=times;g.create_dataset('pressure_perturbation_pa',data=pressure,compression='gzip');g.create_dataset('horizontal_wind_away_m_s',data=wind,compression='gzip');g.attrs['source_archive']=str(root.resolve());g.attrs['source_hashes_json']=json.dumps(hashes);g.attrs['mode_identification']='Not established by this export';print(name,len(times),pressure.shape,flush=True)
 out.attrs['complete']=1
