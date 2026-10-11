"""Closed-run pressure time/altitude diagnostic near a requested geographic site."""
from pathlib import Path
import argparse,hashlib
import numpy as np,h5py
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--lat',type=float,default=42.619);p.add_argument('--lon',type=float,default=-71.491);p.add_argument('--name',default='Millstone Hill');a=p.parse_args();assert(a.run/'run.closed').is_file();a.output.mkdir(parents=True,exist_ok=True)
with h5py.File(a.run/'mesh.h5')as f:
 v=f['vertices_unit_xyz'][:].reshape(-1,3);tri=f['triangle_vertex_indices'][:].reshape(-1,3).astype(int);centers=v[tri].sum(1);centers/=np.linalg.norm(centers,axis=1)[:,None]
phi,lam=np.deg2rad([a.lat,a.lon]);point=np.array([np.cos(phi)*np.cos(lam),np.cos(phi)*np.sin(lam),np.sin(phi)]);i=int(np.argmax(centers@point));actual_lat=np.rad2deg(np.arcsin(centers[i,2]));actual_lon=np.rad2deg(np.arctan2(centers[i,1],centers[i,0]));offset=6371*np.arccos(np.clip(centers[i]@point,-1,1))
for path in a.run.glob('rank_*.h5'):
 with h5py.File(path)as f:
  assert f.attrs['run_complete']==1;ids=f['global_column_id'][:].astype(int)
  if i not in ids:continue
  n=int(np.where(ids==i)[0][0]);z=f['altitude_m'][:];faces=f['altitude_faces_m'][:];nz=len(z);R=f['background_gas_constant_j_kg_k'][:];cv=f['background_cv_j_kg_k'][:];p0=f['column_reference_pressure_pa'][:].reshape(-1,nz)[n] if 'column_reference_pressure_pa'in f else f['background_pressure_pa'][:];times=[];pressures=[];temperature=[];radial=[]
  for key in sorted(f['frames']):
   g=f['frames'][key];q=g['conserved_density_ecef_momentum_energy_ch4_o2'][:].reshape(-1,nz,7)[n];assert np.isfinite(q).all();vel=q[:,1:4]/q[:,0,None];internal=q[:,4]-.5*q[:,0]*(vel*vel).sum(-1);T=internal/q[:,0]/cv;assert(T>0).all();pressures.append(q[:,0]*R*T);temperature.append(T);radial.append(vel@centers[i]);times.append(g.attrs['time_s'])
  source=path;break
else:raise ValueError('Column absent')
t=np.array(times);pressure=np.array(pressures);dp=100*(pressure/p0-1);dt=np.diff(t);np.testing.assert_allclose(dt,dt[0]);window=int(round(7200/dt[0]));window+=1-window%2;assert window<=len(t);half=window//2
trend=savgol_filter(dp,window,2,axis=0);osc=dp-trend;osc[:half]=np.nan;osc[-half:]=np.nan
with h5py.File(a.output/'pressure_time_altitude.h5','w')as f:
 f.attrs.update(site=a.name,requested_latitude_deg=a.lat,requested_longitude_deg=a.lon,cell_latitude_deg=actual_lat,cell_longitude_deg=actual_lon,cell_offset_km=offset,global_cell_id=i,trend_method='centered Savitzky-Golay quadratic, 2-hour window; unsupported edges NaN',trend_window_samples=window,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 for key,value in dict(time_s=t,altitude_m=z,pressure_pa=pressure,pressure_departure_percent=dp,oscillatory_pressure_percent=osc,temperature_k=temperature,radial_velocity_m_s=radial).items():f.create_dataset(key,data=value)
plt.rcParams.update({'font.size':11,'figure.facecolor':'white'})
fig,axs=plt.subplots(1,2,figsize=(11,5),sharey=True,layout='constrained');tf=np.r_[t[0]-(t[1]-t[0])/2,(t[:-1]+t[1:])/2,t[-1]+(t[-1]-t[-2])/2]/3600
for ax,x,title in zip(axs,[dp,osc],['Pressure departure from reference','Oscillations · 2-hour trend removed']):
 peak=max(np.nanmax(abs(x)),1e-10);norm=AsinhNorm(linear_width=peak*.03,vmin=-peak,vmax=peak);m=ax.pcolormesh(tf,faces/1000,x.T,cmap='plasma',norm=norm,rasterized=True);cb=fig.colorbar(m,ax=ax,pad=.02);cb.set_label('Pressure departure [%]');ax.set(title=title,xlabel='Elapsed time [h]',xlim=(0,t[-1]/3600),ylim=(0,faces[-1]/1000));ax.axvline(6,color='#333',lw=.7,ls='--')
axs[0].set_ylabel('Altitude [km]');fig.suptitle(f'{a.name} vicinity · grid cell {actual_lat:.1f}°N, {abs(actual_lon):.1f}°{'W' if actual_lon<0 else 'E'}\nNearest cell centre {offset:.0f} km from site · IAU ends at 6 h',fontsize=12);fig.savefig(a.output/'pressure_time_altitude.png',dpi=180);fig.savefig(a.output/'pressure_time_altitude.pdf');plt.close(fig)
print('Cell',i,'offset km',offset,'raw peak %',abs(dp).max(),'oscillation peak %',np.nanmax(abs(osc)))
