"""Crop the prescribed ERA5-MSIS reference at an existing physical face."""
from pathlib import Path
import argparse,hashlib,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--top-km',type=float,default=450);p.add_argument('--output',default='data/era5_msis_transport_450km_control.h5');a=p.parse_args()
source=Path('data/era5_msis_transport_20250306_23utc.h5');out=Path(a.output);assert not out.exists()
with h5py.File(source) as f,h5py.File(out,'w') as g:
 faces=f['altitude_faces_m'][:];ix=np.flatnonzero(abs(faces-a.top_km*1000)<1e-6);assert len(ix)==1;n=int(ix[0]);nz=len(faces)-1
 for key in f:f.copy(key,g)
 for key in list(g):
  v=g[key]
  if isinstance(v,h5py.Dataset) and v.shape in [(nz,),(nz+1,)]:
   x=v[:n+(key=='altitude_faces_m')];attrs=dict(v.attrs);del g[key];ds=g.create_dataset(key,data=x);ds.attrs.update(attrs)
 for key in list(g['transport']):
  v=g['transport'][key]
  if v.shape and v.shape[0]==nz:
   x=v[:n];attrs=dict(v.attrs);del g['transport'][key];ds=g['transport'].create_dataset(key,data=x);ds.attrs.update(attrs)
 g.attrs.update(dict(f.attrs));g.attrs.update(control_top_km=a.top_km,parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),crop_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),purpose='450 km upper-domain control; original reference retained')
 print(n,'layers')
