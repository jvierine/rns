//! Native Metal float32 for MUSCL reconstruction and shared HLLE face fluxes.
use crate::{mesh::*,parallel::Domain,physics::*,solver::Model};
use anyhow::{Result,ensure};
use std::{ffi::{c_void,CString},ptr::NonNull};
unsafe extern "C" {
 fn rns_metal_new(source:*const i8)->*mut c_void;
 fn rns_metal_release(ptr:*mut c_void);
 fn rns_metal_buffer(ctx:*mut c_void,bytes:usize)->*mut c_void;
 fn rns_metal_contents(ptr:*mut c_void)->*mut c_void;
 fn rns_metal_run(ctx:*mut c_void,buffers:*mut *mut c_void,cells:u32,nz:u32,faces:u32)->bool;
 fn rns_metal_flux(ctx:*mut c_void,input:*mut c_void,output:*mut c_void,count:u32)->bool;
}
struct Buffer(NonNull<c_void>);
impl Drop for Buffer {fn drop(&mut self){unsafe{rns_metal_release(self.0.as_ptr());}}}
impl Buffer {
 fn new(ctx:NonNull<c_void>,bytes:usize)->Result<Self>{Ok(Self(NonNull::new(unsafe{rns_metal_buffer(ctx.as_ptr(),bytes.max(4))}).ok_or_else(||anyhow::anyhow!("Metal allocation failed"))?))}
 fn write<T:Copy>(&self,x:&[T]){unsafe{std::ptr::copy_nonoverlapping(x.as_ptr(),rns_metal_contents(self.0.as_ptr()).cast::<T>(),x.len());}}
 fn read(&self,n:usize)->Vec<f32>{unsafe{std::slice::from_raw_parts(rns_metal_contents(self.0.as_ptr()).cast::<f32>(),n).to_vec()}}
}
pub struct NativeMetal {ctx:NonNull<c_void>,plan:Option<Plan>}
impl Drop for NativeMetal {fn drop(&mut self){unsafe{rns_metal_release(self.ctx.as_ptr());}}}
impl NativeMetal {
 pub fn new()->Result<Self>{let source=CString::new(include_str!("metal/muscl.metal"))?;let ctx=NonNull::new(unsafe{rns_metal_new(source.as_ptr())}).ok_or_else(||anyhow::anyhow!("Native Metal unavailable"))?;Ok(Self{ctx,plan:None})}
 pub fn model_fluxes_into(&mut self,m:&Model,d:&Domain,u:&[Q],columns:&[usize],edges:&[usize],horizontal:&mut[Q],vertical:&mut Vec<Q>)->Result<()> {
 if self.plan.is_none(){self.plan=Some(Plan::new(self.ctx,m,d,columns,edges)?);}
 let p=self.plan.as_ref().unwrap();
 // The shared input buffer is reused after the previous command completed.
 // Pack directly into it; avoid a second full-size allocation and memcpy.
 let values=unsafe{std::slice::from_raw_parts_mut(rns_metal_contents(p.buffers[0].0.as_ptr()).cast::<f32>(),u.len()*18)};
 let pack=|idx:usize,q:Q,bg:Atmos|-> [f32;9] {
    let resting=q==base(bg);let temperature=m.temperature_at(idx%p.nz) && m.second_order;
    if resting {return [0.,0.,0.,0.,0.,0.,0.,(1+2*temperature as u8)as f32,0.];}
    let v=primitive(q,bg);
    [ (v[0]-1.)as f32,(v[1]/1000.)as f32,(v[2]/1000.)as f32,(v[3]/1000.)as f32,
      ((if temperature{v[4]/v[0]}else{v[4]})-1.)as f32,v[5]as f32,v[6]as f32,
      (2*temperature as u8)as f32,(v[4]-1.)as f32 ]
 };
 for(idx,(&q,out))in u.iter().zip(values.chunks_exact_mut(18)).enumerate(){
    let value=pack(idx,q,m.background[idx%p.nz]);out[..9].copy_from_slice(&value);
    if m.column_background.is_none(){out[9..].copy_from_slice(&value);}
    else{out[9..].copy_from_slice(&pack(idx,q,m.at(d.needed[idx/p.nz],idx%p.nz)));}
 }
 let mut ptrs=p.buffers.iter().map(|b|b.0.as_ptr()).collect::<Vec<_>>();
 ensure!(unsafe{rns_metal_run(self.ctx.as_ptr(),ptrs.as_mut_ptr(),u.len()as u32,p.nz as u32,p.scales.len()as u32)},"Metal execution failed");
 let output=unsafe{std::slice::from_raw_parts(rns_metal_contents(p.buffers[8].0.as_ptr()).cast::<f32>(),p.scales.len()*7)};
 let horizontal_count=horizontal.len();
 ensure!(p.scales.len()>=horizontal_count,"Flux output layout");
 vertical.resize(p.scales.len()-horizontal_count,[0.;7]);
 for (idx,(q,s))in output.chunks_exact(7).zip(&p.scales).enumerate(){
    let value=std::array::from_fn(|f|q[f]as f64*s[if f==4{2}else if f==0||f>=5{0}else{1}]);
    ensure!(value.iter().all(|x|x.is_finite()),"Nonfinite Metal flux");
    if idx<horizontal_count{horizontal[idx]=value;}else{vertical[idx-horizontal_count]=value;}
 }
 Ok(())
 }
 pub fn fluxes(&self,packed:&[f32])->Result<Vec<f32>>{let n=packed.len()/24;let i=Buffer::new(self.ctx,packed.len()*4)?;let o=Buffer::new(self.ctx,n*7*4)?;i.write(packed);ensure!(unsafe{rns_metal_flux(self.ctx.as_ptr(),i.0.as_ptr(),o.0.as_ptr(),n as u32)},"Metal flux execution failed");Ok(o.read(n*7))}
}
struct Plan {buffers:Vec<Buffer>,scales:Vec<[f64;3]>,nz:usize}
impl Plan {
 fn new(ctx:NonNull<c_void>,m:&Model,d:&Domain,columns:&[usize],edges:&[usize])->Result<Self> {
        let nz=m.background.len();let nc=d.needed.len();
        let mut topology=vec![0u32;nc*4];let mut geom=vec![0.;nc*18];
        for (slot,&i) in d.needed.iter().enumerate() {
            if columns.binary_search(&i).is_err(){continue;}
            topology[slot*4+3]=1;
            let up=m.mesh.cells[i].center;let axis=if up[2].abs()<0.9{[0.,0.,1.]}else{[1.,0.,0.]};
            let e=unit(cross(axis,up));let n=cross(up,e);
            let neighbors=m.mesh.neighbors(i);
            let mut aa=0.;let mut b=0.;let mut c=0.;let mut xy=vec![];
            for j in neighbors {
                let delta=sub(m.mesh.cells[j].center,up);let x=dot(delta,e);let y=dot(delta,n);let w=1./(x*x+y*y);
                aa+=w*x*x;b+=w*x*y;c+=w*y*y;xy.push((j,delta,x,y,w));
            }
            let det=aa*c-b*b;
            for (j,(neighbor,delta,x,y,w)) in xy.iter().enumerate() {
                topology[slot*4+j]=d.dense_slots[*neighbor] as u32;
                let coefficient=add(mul(e,(c*w*x-b*w*y)/det),mul(n,(aa*w*y-b*w*x)/det));
                geom[slot*18+j*3..slot*18+j*3+3].copy_from_slice(&coefficient);
                geom[slot*18+9+j*3..slot*18+12+j*3].copy_from_slice(&mul(*delta,0.5));
            }
        }
        let mut vert=vec![];
        for k in 0..nz {vert.extend_from_slice(&[if k>0{m.vertical.centers[k]-m.vertical.centers[k-1]}else{1.},if k+1<nz{m.vertical.centers[k+1]-m.vertical.centers[k]}else{1.},m.vertical.widths[k]]);}
        let mut face_ids=vec![];let mut fg=vec![];let mut scales=vec![];
        let rch=1.380649e-23/(16.043*1.66053906660e-27);let ro2=1.380649e-23/(31.998*1.66053906660e-27);
        let mut push=|left:usize,right:Option<usize>,al:Atmos,ar:Atmos,normal:V,dl:V,dr:V,vl:f64,vr:f64,mirror_l:bool,mirror_r:bool,fraction:f64| {
            let (_,_,af)=face_states(base(al),base(ar),al,ar,fraction);
            let gamma=1.+af.r/af.cv;let cs=(gamma*af.p/af.rho).sqrt();let en=af.p*af.cv/af.r;
            face_ids.extend_from_slice(&[left as u32,right.map(|x|x as u32).unwrap_or(u32::MAX),mirror_l as u32,mirror_r as u32,(dl==[0.;3] && dr==[0.;3])as u32]);
            fg.extend_from_slice(&[normal[0],normal[1],normal[2],dl[0],dl[1],dl[2],dr[0],dr[1],dr[2],vl,vr,gamma,rch/af.r,ro2/af.r,3.*rch/af.cv,2.5*ro2/af.cv,1000./cs,if m.second_order {if m.reconstruct_temperature {2.}else{1.}}else{0.}]);
            scales.push([af.rho*cs,af.rho*cs*cs,en*cs]);
        };
        for &eid in edges {
            let e=&m.mesh.edges[eid];let dl=mul(sub(m.mesh.cells[e.right].center,m.mesh.cells[e.left].center),0.5);
            for k in 0..nz {push(d.index(e.left,k,nz),Some(d.index(e.right,k,nz)),m.background[k],m.background[k],e.outward,dl,mul(dl,-1.),0.,0.,false,false,0.5);}
        }
        for &i in &d.owned {
            let normal=unit(m.mesh.cells[i].vector_area);let first=d.index(i,0,nz);let a=m.at(i,0);
            push(first,Some(first),a,a,normal,[0.;3],[0.;3],-0.5,-0.5,true,false,0.5);
            for k in 0..nz {
                let q=d.index(i,k,nz);let a=m.at(i,k);
                if k+1<nz {push(q,Some(q+1),a,m.at(i,k+1),normal,[0.;3],[0.;3],0.5,-0.5,false,false,m.vertical.widths[k]/(m.vertical.widths[k]+m.vertical.widths[k+1]));}
                else if m.reflect_top {push(q,Some(q),a,a,normal,[0.;3],[0.;3],0.5,0.5,false,true,0.5);}
                else {push(q,None,a,a,normal,[0.;3],[0.;3],0.5,0.,false,false,0.5);}
            }
        }

 let mut buffers=Vec::new();
 let floats=|v:&[f64]|->Result<Buffer>{let b=Buffer::new(ctx,v.len()*4)?;b.write(&v.iter().map(|x|*x as f32).collect::<Vec<_>>());Ok(b)};
 let uints=|v:&[u32]|->Result<Buffer>{let b=Buffer::new(ctx,v.len()*4)?;b.write(v);Ok(b)};
 buffers.push(Buffer::new(ctx,nc*nz*18*4)?);buffers.push(uints(&topology)?);buffers.push(floats(&geom)?);buffers.push(floats(&vert)?);buffers.push(Buffer::new(ctx,nc*nz*28*4)?);buffers.push(uints(&face_ids)?);buffers.push(floats(&fg)?);buffers.push(Buffer::new(ctx,scales.len()*24*4)?);buffers.push(Buffer::new(ctx,scales.len()*7*4)?);
 Ok(Self{buffers,scales,nz})
 }
}
