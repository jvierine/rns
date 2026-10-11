"""Globe and meridional animations from a closed ERA5 viewer layer."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter
from matplotlib.colors import AsinhNorm
p=argparse.ArgumentParser();p.add_argument('layer',type=Path);p.add_argument('--fps',type=int,default=8);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--world',default='/Users/jvi019/src/heimdall-production/web/public/world.geojson');a=p.parse_args();assert(a.run/'run.closed').is_file();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((a.layer/'metadata.json').read_text());nc=m['cell_count'];nz=len(m['altitude_m']);z=np.array(m['altitude_m']);faces=np.array(m['altitude_faces_m'])/1000;times=np.array(m['times_s']);xy=np.fromfile(a.layer/'geometry.f32',dtype='<f4').reshape(nc,4,2)[:,:3];centers=np.fromfile(a.layer/'centers.f32',dtype='<f4').reshape(nc,3)
lon=np.deg2rad(xy[:,:,0]);lat=np.deg2rad(xy[:,:,1]);verts=np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],-1)
with h5py.File(a.run/'rank_0000.h5')as f:p0=f['background_pressure_pa'][:]
latitudes=np.linspace(-90,90,121);angle=np.deg2rad(latitudes);meridian=-30.;l=np.deg2rad(meridian);points=np.stack([np.cos(angle)*np.cos(l),np.cos(angle)*np.sin(l),np.sin(angle)],-1);ids=np.argmax(points@centers.T,axis=1)
world=json.loads(Path(a.world).read_text());rings=[]
for f in world['features']:
 if f['geometry']['type']=='Polygon':rings.extend(f['geometry']['coordinates'])
 elif f['geometry']['type']=='MultiPolygon':
  for polygon in f['geometry']['coordinates']:rings.extend(polygon)
plt.rcParams.update({'font.size':11,'figure.facecolor':'white'})
for field,label,units,height in [('pressure_perturbation','Pressure relative to reference atmosphere','%',1000),('up','Upward velocity','m/s',21000)]:
 values=np.stack([np.fromfile(a.layer/field/f'{i:04d}.f32',dtype='<f4').reshape(nz,nc) for i in range(len(times))]).astype(float)
 if field=='pressure_perturbation':
  reference=np.fromfile(a.layer/'reference_pressure.f32',dtype='<f4').reshape(nz,nc) if (a.layer/'reference_pressure.f32').is_file() else p0[:,None]
  values=100*values/reference[None,:,:]
 k=int(np.argmin(abs(z-height)));surface=values[:,k];section=values[:,:,ids];peak=max(abs(surface).max(),1e-10);cutpeak=max(abs(section).max(),1e-10)
 norm=AsinhNorm(linear_width=peak*.03,vmin=-peak,vmax=peak);cutnorm=AsinhNorm(linear_width=cutpeak*.03,vmin=-cutpeak,vmax=cutpeak)
 fig=plt.figure(figsize=(12,5.8));gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.25],left=.015,right=.91,bottom=.25,top=.84,wspace=.08);axes=[fig.add_subplot(gs[0,j])for j in range(3)];collections=[]
 for ax,longitude,title in zip(axes[:2],[-30.,150.],['Atlantic / Americas','Pacific / Asia']):
  ll=np.deg2rad(longitude);view=np.array([np.cos(ll),np.sin(ll),0]);east=np.array([-np.sin(ll),np.cos(ll),0]);north=np.array([0,0,1]);mask=centers@view>0
  polys=np.stack([verts@east,verts@north],-1)[mask];collection=PolyCollection(polys,cmap='plasma',norm=norm,edgecolors='none',antialiased=False);collection.set_array(surface[0,mask]);ax.add_collection(collection);collections.append((collection,mask))
  ax.add_patch(plt.Circle((0,0),1,fill=False,color='#777',lw=.6))
  for ring in rings:
   llr,br=np.deg2rad(np.array(ring).T);v=np.stack([np.cos(br)*np.cos(llr),np.cos(br)*np.sin(llr),np.sin(br)],-1);x=v@east;y=v@north;x[v@view<0]=np.nan;ax.plot(x,y,color='#333',lw=.35,alpha=.65)
  ax.set(xlim=(-1.03,1.03),ylim=(-1.03,1.03),aspect='equal',title=title);ax.axis('off')
 ax=axes[2];mesh=ax.pcolormesh(np.linspace(-90.75,90.75,122),faces,section[0],cmap='plasma',norm=cutnorm,rasterized=True);ax.set(xlim=(-90,90),ylim=(0,faces[-1]),xlabel='Latitude [°]',ylabel='Altitude [km]',title='Meridional cut · 30°W');ax.yaxis.tick_right();ax.yaxis.set_label_position('right');ax.set_xticks([-90,-45,0,45,90])
 cb=fig.colorbar(collections[0][0],cax=fig.add_axes([.08,.14,.48,.025]),orientation="horizontal");cb.set_ticks([-peak,-peak*.1,0,peak*.1,peak],labels=[f"{v:.2g}" for v in [-peak,-peak*.1,0,peak*.1,peak]]);cb.set_label(f"Pressure perturbation [%] · {z[k]/1000:g} km" if field=="pressure_perturbation" else f"Upward velocity [m/s] · {z[k]/1000:g} km");cb2=fig.colorbar(mesh,cax=fig.add_axes([.67,.14,.25,.025]),orientation="horizontal");cb2.set_ticks([-cutpeak,0,cutpeak],labels=[f"{v:.2g}" for v in [-cutpeak,0,cutpeak]]);cb2.set_label(f"Pressure perturbation [%]" if field=="pressure_perturbation" else "Upward velocity [m/s]")
 title=fig.suptitle('',fontsize=14)
 def update(i):
  for c,mask in collections:c.set_array(surface[i,mask])
  mesh.set_array(section[i].ravel());stamp=datetime.fromtimestamp(m['reference_unix_s']+times[i],timezone.utc).strftime('%Y-%m-%d %H:%M UTC');title.set_text(f'{"ERA5–MSIS" if faces[-1]>40 else "ERA5"} · {label}\n{stamp}');return [mesh,*[c for c,_ in collections]]
 animation=FuncAnimation(fig,update,frames=len(times),interval=125);animation.save(a.output/f'{field}.mp4',writer=FFMpegWriter(fps=a.fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20']),dpi=110);animation.save(a.output/f'{field}.gif',writer=PillowWriter(fps=a.fps),dpi=65);update(len(times)-1);fig.savefig(a.output/f'{field}_last.png',dpi=130);plt.close(fig)
 with h5py.File(a.output/f'{field}_display_validation.h5','w')as f:
  f.attrs.update(field=field,units=units,globe_altitude_m=z[k],cut_longitude_deg=meridian,globe_limit=peak,cut_limit=cutpeak,transform='signed asinh',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());f.create_dataset('time_s',data=times);f.create_dataset('globe_peak',data=abs(surface).max(1));f.create_dataset('cut_peak',data=abs(section).max(axis=(1,2)))
 print(field,len(times),'frames')
