"""Plot closed outgoing/returned acoustic characteristics from Rust boundary audit."""
from pathlib import Path
import hashlib
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('data/top_absorption_audit.h5');out=Path('output/top_absorption_audit');out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'figure.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True,layout='constrained')
with h5py.File(p,'r+')as f:
 assert f.attrs['complete']==1
 f.attrs['setup']='Uniform 300 K gas, zero gravity, spherical radial acoustic pulse; 2 km layers, top 300 km, probe 179 km, pulse centred at 100 km, sigma 8 km; second-order Rust CPU'
 f.attrs['energy_ratio_definition']='Integral incoming characteristic squared at 500-1400 s / outgoing squared at 0-450 s; not full energy budget'
 for path in ['src/bin/test_top_absorption.rs','src/solver.rs','src/physics.rs','scripts/plot_top_absorption.py']:f.attrs[path.replace('/','_')+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 for ax,name,title in zip(axs.flat,['wall','wall_sponge','open','open_sponge'],['Reflecting lid','Reflecting lid + sponge','Open top','Open top + sponge']):
  g=f[name];r=g['time_outgoing_incoming_pa'][:].reshape(-1,3);scale=abs(r[:,1]).max();ax.plot(r[:,0],r[:,1]/scale,label='Outgoing',lw=1.3);ax.plot(r[:,0],r[:,2]/scale,label='Returned',lw=1.3);ratio=g.attrs['reflected_characteristic_energy_ratio'];ax.set_title(f'{title} · returned {100*ratio:.4g}%');ax.grid(alpha=.2);ax.set_xlim(0,1400)
for ax in axs[1]:ax.set_xlabel('Time [s]')
fig.supylabel('Characteristic pressure / outgoing peak')
axs[0,0].legend();fig.suptitle('Top-boundary acoustic reflection test');fig.savefig(out/'reflection.png',dpi=160);fig.savefig(out/'reflection.pdf');plt.close(fig)
