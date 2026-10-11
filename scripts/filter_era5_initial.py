"""Spherical low-pass of ERA5 initial fields; no filtering during evolution."""
import argparse,hashlib,shutil
from pathlib import Path
import h5py,numpy as np
from scipy.special import sph_harm_y
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--degree',type=int,default=6);p.add_argument('--width-km',type=float,default=1500);a=p.parse_args();assert a.degree>=1 and a.width_km>=0 and not a.output.exists()
with h5py.File(a.input)as f:
 assert f.attrs['complete']==1;lat=f['latitude_deg'][:];lon=f['longitude_deg'][:];fields={n:f['initial/'+n][:] for n in ['pressure_pa','temperature_k','east_m_s','north_m_s','up_m_s']}
 b=np.r_[-90,(lat[:-1]+lat[1:])/2,90];weights=np.repeat(np.diff(np.sin(np.deg2rad(b)))/2,len(lon))/len(lon)
 la,lo=np.meshgrid(np.deg2rad(lat),np.deg2rad(lon),indexing='ij');theta=np.pi/2-la;B=[];degrees=[]
 for l in range(a.degree+1):
  for m in range(-l,l+1):
   y=sph_harm_y(l,abs(m),theta,lo);B.append((y.real if m==0 else np.sqrt(2)*(y.real if m>0 else y.imag)).ravel());degrees.append(l)
 B=np.stack(B,1);degree=np.array(degrees);gram=B.T@(weights[:,None]*B);atten=np.exp(-.5*degree*(degree+1)*(a.width_km*1000/6371008.8)**2)
 def filt(x):
  shape=x.shape;v=x.reshape(shape[0],-1).T;c=np.linalg.solve(gram,B.T@(weights[:,None]*v));return (B@(atten[:,None]*c)).T.reshape(shape)
 up=np.stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)],-1);east=np.stack([-np.sin(lo),np.cos(lo),np.zeros_like(lo)],-1);north=np.cross(up,east)
 result={}
 for name in ['pressure_pa','temperature_k']:
  result[name]=np.exp(filt(np.log(fields[name])))
  original=(fields[name].reshape(len(fields[name]),-1)*weights).sum(1);new=(result[name].reshape(len(fields[name]),-1)*weights).sum(1);result[name]*=(original/new)[:,None,None]
 wind=fields['east_m_s'][...,None]*east+fields['north_m_s'][...,None]*north
 filtered=np.stack([filt(wind[...,i]) for i in range(3)],-1);filtered-=(filtered*up).sum(-1)[...,None]*up
 result['east_m_s']=(filtered*east).sum(-1);result['north_m_s']=(filtered*north).sum(-1);result['up_m_s']=np.zeros_like(fields['up_m_s']);result['density_kg_m3']=result['pressure_pa']/(287*result['temperature_k'])
 # Operator checks: constants remain constant; a retained harmonic is multiplied
 # by precisely the selected spectral weight. Wind remains tangent to the sphere.
 np.testing.assert_allclose(filt(np.ones((1,len(lat),len(lon)))),1,atol=2e-14)
 j=len(degrees)-1;np.testing.assert_allclose(filt(B[:,j].reshape(1,len(lat),len(lon))),atten[j]*B[:,j].reshape(1,len(lat),len(lon)),atol=2e-14)
 assert abs((filtered*up).sum(-1)).max()<1e-12
shutil.copyfile(a.input,a.output)
with h5py.File(a.output,'r+')as f:
 f.attrs['complete']=0
 for name,v in result.items():assert np.isfinite(v).all();f['initial/'+name][...]=v
 # Refresh the existing horizontally averaged background arrays consistently.
 f['pressure_pa'][...]=(result['pressure_pa'].reshape(len(result['pressure_pa']),-1)*weights).sum(1)
 f['density_kg_m3'][...]=(result['density_kg_m3'].reshape(len(result['density_kg_m3']),-1)*weights).sum(1)
 f.attrs.update(initial_filter='real spherical harmonics; log p/T with preserved area means; tangent ECEF wind; vertical wind zero',filter_degree=a.degree,filter_width_km=a.width_km,filter_input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),filter_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),filter_status='spatial projection only; not dynamically balanced; mass/energy may change at initialization',complete=1)
 f.create_dataset('filter/degree',data=degree);f.create_dataset('filter/amplitude',data=atten)
print(a.output,'degree',a.degree,'width km',a.width_km,'Gram condition',np.linalg.cond(gram))
