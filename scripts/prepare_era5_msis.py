"""Join the ERA5 reference column to MSIS thermodynamics on a stretched grid."""
import argparse, hashlib
from pathlib import Path
import h5py, numpy as np, pymsis
from pymsis.utils import get_f107_ap
from scipy.integrate import cumulative_trapezoid
p=argparse.ArgumentParser();p.add_argument('--era5',default='data/era5_20250306_23utc.h5');p.add_argument('--output',default='data/era5_msis_20250306_23utc.h5');a=p.parse_args()
faces=np.r_[np.arange(0,40001,2000),np.arange(45000,100001,5000),np.arange(110000,200001,10000),np.arange(225000,1000001,25000)].astype(float);z=(faces[:-1]+faces[1:])/2
fine=np.unique(np.r_[np.arange(0,1000001,250),z,30000.,50000.]);dates=np.array(['2025-03-06T23:00'],dtype='datetime64[m]');f107,f107a,ap=get_f107_ap(dates)
raw=pymsis.calculate(np.repeat(dates,len(fine)),np.zeros(len(fine)),np.zeros(len(fine)),fine/1000,f107s=np.repeat(f107,len(fine)),f107as=np.repeat(f107a,len(fine)),aps=np.repeat(ap,len(fine),axis=0),version=2.1).reshape(-1,11).astype(float)
n=raw[:,[1,2,3,4,5,6,7,9]];n=np.nan_to_num(n,nan=0.);amu=np.array([28.0134,31.998,15.999,4.0026,1.00794,39.948,14.0067,30.0061]);rho=n@(amu*1.66053906660e-27);R=1.380649e-23*n.sum(1)/rho;cv=1.380649e-23*(n@np.array([2.5,2.5,1.5,1.5,1.5,1.5,1.5,2.5]))/rho
with h5py.File(a.era5) as f:
 ez=f['altitude_m'][:];ep=f['pressure_pa'][:];erho=f['density_kg_m3'][:];et=ep/(287*erho);epoch=f.attrs['reference_unix_s']
anchorT=np.interp(30000,ez,et);anchorP=np.exp(np.interp(30000,ez,np.log(ep)));blend=np.clip((fine-30000)/20000,0,1)
T=(1-blend)*anchorT+blend*raw[:,10];T[fine<=30000]=np.interp(fine[fine<=30000],ez,et)
R=(1-blend)*287+blend*R;cv=(1-blend)*717.5+blend*cv;g=9.81*(6371000/(6371000+fine))**2
integral=cumulative_trapezoid(g/(R*T),fine,initial=0);pressure=anchorP*np.exp(-(integral-np.interp(30000,fine,integral)));pressure[fine<=30000]=np.exp(np.interp(fine[fine<=30000],ez,np.log(ep)))
values=dict(altitude_m=z,altitude_faces_m=faces,pressure_pa=np.exp(np.interp(z,fine,np.log(pressure))),gas_constant_j_kg_k=np.interp(z,fine,R),cv_j_kg_k=np.interp(z,fine,cv),gravity_m_s2=np.interp(z,fine,g));values['density_kg_m3']=values['pressure_pa']/(values['gas_constant_j_kg_k']*np.interp(z,fine,T))
for name,x in values.items():assert np.isfinite(x).all() and (np.all(x>0) if name!='altitude_faces_m' else x[0]==0)
with h5py.File(a.output,'w') as f:
 f.attrs.update(complete=1,reference_unix_s=epoch,msis_version='2.1',msis_latitude_deg=0.,msis_longitude_deg=0.,upper_profile='single equatorial MSIS thermodynamic reference; hydrostatic pressure anchored to ERA5 at 30 km',join='temperature/R/cv blended 30-50 km; ERA5 horizontal fields initialized separately',era5_sha256=hashlib.sha256(Path(a.era5).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 for name,x in values.items():f.create_dataset(name,data=x)
 f.create_dataset('msis_reference/altitude_m',data=fine);f.create_dataset('msis_reference/raw_output',data=raw);f.create_dataset('drivers/f107',data=f107);f.create_dataset('drivers/f107a',data=f107a);f.create_dataset('drivers/ap',data=ap)
print(a.output,len(z),'layers',faces[-1]/1000,'km','F107',f107,'F107a',f107a,'Ap',ap)
