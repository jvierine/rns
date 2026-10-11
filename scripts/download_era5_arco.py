"""One ERA5 hour from the public ARCO archive, sampled to 2.5 degrees."""
import argparse, json, hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests, numcodecs, numpy as np, h5py
p=argparse.ArgumentParser();p.add_argument('--time',default='2025-03-06T23:00:00');p.add_argument('--output',default='data/era5_20250306_23utc.h5');a=p.parse_args()
base='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
meta=requests.get(base+'/.zmetadata',timeout=30).json()['metadata']
index=int((np.datetime64(a.time)-np.datetime64('1940-01-01T00:00:00'))/np.timedelta64(1,'h'))
def get(name,key):
 r=requests.get(base+'/'+name+'/'+key,timeout=180);r.raise_for_status();spec=meta[name+'/.zarray'];raw=numcodecs.get_codec(spec['compressor']).decode(r.content) if spec['compressor'] else r.content
 return np.frombuffer(raw,dtype=spec['dtype']).reshape(spec['chunks']),hashlib.sha256(r.content).hexdigest()
def field(name):
 v,sha=get(name,f'{index}.0.0.0');v=v[0,:,::10,::10].copy();print('Fetched',name,flush=True);return name,v,sha
names=['temperature','geopotential','u_component_of_wind','v_component_of_wind','vertical_velocity']
with ThreadPoolExecutor(max_workers=3) as pool: result=list(pool.map(field,names))
lat,_=get('latitude','0');lon,_=get('longitude','0');level,_=get('level','0')
time_spec=meta['time/.zarray'];timechunk,_=get('time',str(index//time_spec['chunks'][0]));record=timechunk[index%time_spec['chunks'][0]]
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.output,'w')as f:
 f.attrs.update(schema='rns.era5.pressure-levels/v1',time_utc=a.time,source=base,era5_time_coordinate_value=str(record),era5_time_units=meta['time/.zattrs'].get('units',''),reference_unix_s=datetime.fromisoformat(a.time).replace(tzinfo=timezone.utc).timestamp(),download_complete=1,horizontal_sampling_deg=2.5)
 f.create_dataset('latitude_deg',data=lat[::10]);f.create_dataset('longitude_deg',data=lon[::10]);f.create_dataset('pressure_pa',data=level*100.)
 for name,v,sha in result:
  assert np.isfinite(v).all(),name
  d=f.create_dataset(name,data=v,compression='gzip',shuffle=True);d.attrs['chunk_sha256']=sha;d.attrs['units']=meta[name+'/.zattrs'].get('units','')
print('Saved',a.output,flush=True)
