"""Animate actual Rust/MPI great-circle sections; fixed physical colour scales."""
from pathlib import Path
import argparse,json,hashlib
import h5py,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
with h5py.File(a.input) as f:
 assert f.attrs['complete']==1;x=f['distance_km'][:];peru=f.attrs['jicamarca_distance_km'];cases={n:{k:f[n+'/'+k][:] for k in ['time_s','altitude_km','horizontal_wind_away_m_s','pressure_perturbation_pa']} for n in ['lamb','mesosphere']}
times=cases['mesosphere']['time_s'];limits=[float(abs(cases[n]['horizontal_wind_away_m_s']).max())*1000 for n in cases]
# Standalone mesospheric pressure/wind example, retaining actual saved cadence.
c=cases['mesosphere'];fig,axes=plt.subplots(2,1,figsize=(10,6.5),layout='constrained');meshes=[]
for i,(quantity,units,mult,label) in enumerate([('pressure_perturbation_pa','Pressure perturbation [mPa]',1000,'Pressure'),('horizontal_wind_away_m_s','Wind away from Tonga [mm/s]',1000,'Horizontal wind')]):
 v=c[quantity]*mult;lim=float(abs(v).max());mesh=axes[i].pcolormesh(x,c['altitude_km'],v[0].T,cmap='RdBu_r',norm=SymLogNorm(lim*.015,vmin=-lim,vmax=lim),shading='nearest');meshes.append((mesh,quantity,mult));axes[i].set(xlim=(0,12000),ylim=(0,200),ylabel='Altitude [km]',title=f'{chr(97+i)}) {label}');axes[i].axvline(peru,color='#476856',ls='--',lw=.9);axes[i].axhspan(80,100,color='#888888',alpha=.1);fig.colorbar(mesh,ax=axes[i],pad=.015).set_label(units)
axes[-1].set_xlabel('Great-circle distance from Tonga toward Peru [km]');title=fig.suptitle('MSIS mode experiment · t = 0 h',fontsize=14)
def msis_update(j):
 for mesh,q,mult in meshes:mesh.set_array((c[q][j]*mult).T.ravel())
 title.set_text(f'MSIS heating experiment · t = {times[j]/3600:.2f} h');return [title]
FuncAnimation(fig,msis_update,frames=len(times),interval=180).save(out/'tonga_msis_modes.gif',writer=PillowWriter(fps=6),dpi=90);plt.close(fig)
# Standalone Lamb demo.
c=cases['lamb'];fig,ax=plt.subplots(figsize=(10,4),layout='constrained');v=c['horizontal_wind_away_m_s']*1000;lim=float(abs(v).max())
mesh=ax.pcolormesh(x,c['altitude_km'],v[0].T,cmap='RdBu_r',norm=SymLogNorm(lim*.015,vmin=-lim,vmax=lim),shading='nearest')
ax.set(xlim=(0,12000),ylim=(0,60),xlabel='Great-circle distance from Tonga toward Peru [km]',ylabel='Altitude [km]');ax.axvline(peru,color='#476856',ls='--',lw=.9);fig.colorbar(mesh,ax=ax).set_label('Wind away from Tonga [mm/s]');title=fig.suptitle('Lamb wave')
def lamb_update(j):
 mesh.set_array(v[j].T.ravel());title.set_text(f'Lamb wave · t = {c["time_s"][j]/3600:.2f} h');return [mesh,title]
FuncAnimation(fig,lamb_update,frames=len(c['time_s']),interval=100).save(out/'tonga_lamb.gif',writer=PillowWriter(fps=12),dpi=90);plt.close(fig)
with h5py.File(out/'render_validation.h5','w') as f:
 f['time_s']=times;f['wind_limits_mm_s']=limits;f.attrs['frames']=len(times);f.attrs['input_sha256']=hashlib.sha256(Path(a.input).read_bytes()).hexdigest();f.attrs['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();f.attrs['quantity']='Actual sampled horizontal wind away along the great circle';f.attrs['mode_identity']='Unverified; no imposed L0/L1 speeds or fitted amplitudes';f.attrs['publication_reference']='Poblet et al. 2023 doi:10.1029/2023GL103809'
print('Rendered',len(times),'frames',flush=True)
