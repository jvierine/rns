"""Render only a closed global Rust archive, keeping both hemispheres visible."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter

p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--field',choices=['pressure','mesowind'],default='pressure');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
assert (root/'run.closed').exists()
with h5py.File(root/'mesh.h5') as f:
    vertices=f['vertices_unit_xyz'][:].reshape(-1,3);triangles=f['triangle_vertex_indices'][:].reshape(-1,3).astype(int);centers=f['center_unit_xyz'][:].reshape(-1,3)
    lat=np.deg2rad(f.attrs['source_latitude_deg']);lon=np.deg2rad(f.attrs['source_longitude_deg']);sound=f.attrs['sound_speed_m_s'];amp=f.attrs['initial_surface_pressure_pa']
source=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
tangent=(centers@source)[:,None]*centers-source;tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-15)
values=None;times=None
for path in sorted(root.glob('rank_*.h5')):
    with h5py.File(path) as f:
        assert f.attrs['run_complete']==1
        ids=f['global_column_id'][:].astype(int);keys=sorted(f['frames']);t=np.array([f['frames'][k].attrs['time_s'] for k in keys]);v=np.array([f['frames'][k]['surface_pressure_perturbation_pa'][:] for k in keys]);assert np.isfinite(v).all()
        if a.field=='mesowind':v=np.array([np.sum(f['frames'][k]['mesosphere_80_100km_horizontal_velocity_ecef_m_s'][:].reshape(-1,3)*tangent[ids],axis=1) for k in keys])
        if values is None:values=np.zeros((len(keys),len(centers)));times=t
        assert np.array_equal(t,times);values[:,ids]=v
source=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
east=np.array([-np.sin(lon),np.cos(lon),0]);north=np.cross(source,east)
if a.field=='mesowind':amp=max(float(abs(values).max()),1e-30)
soft=amp*.02
transformed=np.arcsinh(values/soft)/np.arcsinh(amp/soft)
plt.rcParams.update({'font.size':12,'figure.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(10,5.4),layout='constrained');collections=[]
for ax,sign,label in zip(axes,[1,-1],['Pacific / source hemisphere','Opposite hemisphere']):
    mask=centers@source*sign>0
    polygons=[np.column_stack((vertices[tri]@east*sign,vertices[tri]@north)) for tri in triangles[mask]]
    collection=PolyCollection(polygons,cmap='plasma',clim=(-1,1),edgecolors=(0,0,0,.12),linewidths=.12);collection.set_array(transformed[0,mask]);ax.add_collection(collection);collections.append((collection,mask));ax.add_patch(plt.Circle((0,0),1,fill=False,color='black',lw=.6));ax.set(xlim=(-1.03,1.03),ylim=(-1.03,1.03),aspect='equal',title=label);ax.axis('off')
colorbar=fig.colorbar(collections[0][0],ax=axes,orientation='horizontal',fraction=.07,pad=.03)
physical=np.array([-amp,-amp*.1,0,amp*.1,amp]);colorbar.set_ticks(np.arcsinh(physical/soft)/np.arcsinh(amp/soft),labels=[f'{v*1000:.2g}' for v in physical]);colorbar.set_label('Wind away from Tonga [mm/s], mean 80–100 km' if a.field=='mesowind' else 'Surface pressure perturbation [mPa]' )
title=fig.suptitle('')
def update(i):
    for collection,mask in collections:collection.set_array(transformed[i,mask])
    title.set_text(f'Tonga · {times[i]/3600:.2f} h')
    return [c for c,_ in collections]
animation=FuncAnimation(fig,update,frames=len(times),interval=100)
animation.save(out/'global_lamb.mp4',writer=FFMpegWriter(fps=10),dpi=110)
animation.save(out/'global_lamb_iphone.gif',writer=PillowWriter(fps=10),dpi=65)
for i in [0,len(times)//2,len(times)-1]:update(i);fig.savefig(out/f'frame_{i:03d}.png',dpi=140)
with h5py.File(out/'display_validation.h5','w') as f:
    f['time_s']=times;f['surface_peak_abs_pa']=np.max(np.abs(values),axis=1);f['clipped_columns']=np.sum(np.abs(values)>amp,axis=1);f.attrs['archive_complete']=1;f.attrs['scope']='Global isothermal numerical development test, not reconstructed Tonga event.'
print(f'{len(times)} frames, {len(centers)} columns, final time {times[-1]} s')
