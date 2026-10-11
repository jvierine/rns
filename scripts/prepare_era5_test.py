"""Dry ERA5 initialization on fixed heights for the rotating-globe test."""
import argparse,hashlib
from pathlib import Path
import numpy as np,xarray as xr,h5py
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default='data/era5_20250306_23utc.h5');p.add_argument('--layers',type=int,default=20);p.add_argument('--dz',type=float,default=2000.);a=p.parse_args()
d=xr.open_dataset(a.input).isel(valid_time=0).sortby('latitude');lat=d.latitude.values;lon=d.longitude.values;pressure=d.pressure_level.values*100.;height=d.z.values/9.80665
z=(np.arange(a.layers)+.5)*a.dz;shape=(len(z),len(lat),len(lon));values={n:np.empty(shape)for n in ['pressure_pa','temperature_k','east_m_s','north_m_s','up_m_s']}
for j in range(len(lat)):
 for i in range(len(lon)):
  h=height[:,j,i];assert np.all(np.diff(h)>0)
  assert z[-1]<h[-1], 'ERA5 top below requested height'
  # Flat-ground test: pressure-level fields below local terrain are retained as supplied by ERA5.
  values['pressure_pa'][:,j,i]=np.exp(np.interp(z,h,np.log(pressure)))
  for target,src in [('temperature_k','t'),('east_m_s','u'),('north_m_s','v')]: values[target][:,j,i]=np.interp(z,h,d[src].values[:,j,i])
  rho=values['pressure_pa'][:,j,i]/(287.*values['temperature_k'][:,j,i]);values['up_m_s'][:,j,i]=-np.interp(z,h,d.w.values[:,j,i])/(rho*9.80665)
values['density_kg_m3']=values['pressure_pa']/(287.*values['temperature_k'])
with h5py.File(a.output,'w')as f:
 f.attrs.update(schema='rns.era5.initial/v1',reference_unix_s=float(d.valid_time.values.astype('datetime64[s]').astype('int64')),time_utc=str(d.valid_time.values),source='Copernicus ERA5 hourly pressure levels',source_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),vertical_velocity_conversion='w=-omega/(rho*g), dry ideal gas',flat_ground=True,dry=True)
 for n,v in values.items():assert np.isfinite(v).all();f.create_dataset('initial/'+n,data=v,compression='gzip',shuffle=True)
 f.create_dataset('latitude_deg',data=lat);f.create_dataset('longitude_deg',data=lon);f.create_dataset('altitude_m',data=z);f.create_dataset('altitude_faces_m',data=np.arange(a.layers+1)*a.dz)
 weights=np.cos(np.deg2rad(lat));weights[[0,-1]]=0;weights=weights[None,:,None]*np.ones((1,1,len(lon)));weights/=weights.sum()
 mean=lambda v:(v*weights).sum(axis=(1,2))
 f.create_dataset('pressure_pa',data=mean(values['pressure_pa']));f.create_dataset('density_kg_m3',data=mean(values['density_kg_m3']));
 for n,x in [('gas_constant_j_kg_k',287.),('cv_j_kg_k',717.5),('gravity_m_s2',9.80665)]:f.create_dataset(n,data=np.full(len(z),x))
 f.attrs['complete']=1
print(a.output)
