//! Float32 transport matrix application; float64 iterative refinement is in transport.rs.
use super::{Quadrature,Penalty};
use crate::{mesh::*,physics::Q,solver::Model,parallel::Domain};
use anyhow::{Result,ensure};
use std::{ffi::{CString,c_void},ptr::NonNull};
unsafe extern "C" {
 fn rns_transport_new(source:*const i8)->*mut c_void;
 fn rns_transport_cg(ctx:*mut c_void,buffers:*mut *mut c_void,rows:u32,quads:u32,penalties:u32,thermal:u32,nz:u32,line_enabled:u32,dt:f32,tolerance:f32,iterations:*mut u32,residual:*mut f64)->bool;
 fn rns_transport_run(ctx:*mut c_void,buffers:*mut *mut c_void,rows:u32,quads:u32,penalties:u32,thermal:u32,dt:f32)->bool;
 fn rns_metal_release(ptr:*mut c_void);
 fn rns_metal_buffer(ctx:*mut c_void,bytes:usize)->*mut c_void;
 fn rns_metal_contents(ptr:*mut c_void)->*mut c_void;
}
struct Buffer(NonNull<c_void>);
impl Drop for Buffer{fn drop(&mut self){unsafe{rns_metal_release(self.0.as_ptr());}}}
impl Buffer{
 fn new(ctx:NonNull<c_void>,bytes:usize)->Result<Self>{Ok(Self(NonNull::new(unsafe{rns_metal_buffer(ctx.as_ptr(),bytes.max(4))}).ok_or_else(||anyhow::anyhow!("Transport Metal allocation"))?))}
 fn from<T:Copy>(ctx:NonNull<c_void>,data:&[T])->Result<Self>{let b=Self::new(ctx,std::mem::size_of_val(data))?;unsafe{std::ptr::copy_nonoverlapping(data.as_ptr(),rns_metal_contents(b.0.as_ptr()).cast::<T>(),data.len());}Ok(b)}
 fn floats_mut(&mut self,n:usize)->&mut[f32]{unsafe{std::slice::from_raw_parts_mut(rns_metal_contents(self.0.as_ptr()).cast::<f32>(),n)}}
 fn floats(&self,n:usize)->&[f32]{unsafe{std::slice::from_raw_parts(rns_metal_contents(self.0.as_ptr()).cast::<f32>(),n)}}
}
pub struct MetalTransport{buffers:Vec<Buffer>,ctx:NonNull<c_void>,rows:usize,quads:usize,penalties:usize,nz:usize}
impl Drop for MetalTransport{fn drop(&mut self){unsafe{rns_metal_release(self.ctx.as_ptr());}}}
impl MetalTransport{
 pub fn new(quad:&[Quadrature],penalties:&[Penalty],incidence:&[Vec<(usize,V)>],owned:&[bool],mu:&[f64],kappa:&[f64],m:&Model,d:&Domain)->Result<Self>{
 let source=CString::new(include_str!("metal/transport.metal"))?;let ctx=NonNull::new(unsafe{rns_transport_new(source.as_ptr())}).ok_or_else(||anyhow::anyhow!("Transport Metal unavailable"))?;
 let rows=owned.len();let mut qm=vec![];let mut qi=vec![];let mut qc=vec![];let mut qp=vec![];
 for q in quad{qm.extend([q.index as u32,qi.len()as u32,q.pairs.len()as u32]);qp.extend([q.volume as f32,mu[q.layer]as f32,kappa[q.layer]as f32]);for &(n,c)in &q.pairs{qi.push(n as u32);qc.extend(c.map(|v|v as f32));}}
 let mut pm=vec![];let mut pi=vec![];let mut pc=vec![];let mut pw=vec![];let mut pincidence=vec![vec![];rows];
 for (slot,p)in penalties.iter().enumerate(){pm.extend([pi.len()as u32,p.entries.len()as u32]);pw.extend([(p.weight*mu[p.layer])as f32,(p.weight*kappa[p.layer])as f32]);for &(n,b)in &p.entries{pi.push(n as u32);pc.push(b as f32);if owned[n]{pincidence[n].push((slot as u32,b as f32));}}}
 let mut qo=vec![0];let mut qids=vec![];let mut qcoef=vec![];
 for row in incidence{for &(n,c)in row{qids.push(n as u32);qcoef.extend(c.map(|v|v as f32));}qo.push(qids.len()as u32);}
 let mut po=vec![0];let mut pids=vec![];let mut pcoef=vec![];for row in pincidence{for(n,b)in row{pids.push(n);pcoef.push(b);}po.push(pids.len()as u32);}
 ensure!(qi.len()<=u32::MAX as usize && pi.len()<=u32::MAX as usize && rows<=u32::MAX as usize,"Transport index overflow");
 let mut buffers=vec![Buffer::new(ctx,rows*3*4)?,Buffer::new(ctx,rows*4)?,Buffer::from(ctx,&qm)?,Buffer::from(ctx,&qi)?,Buffer::from(ctx,&qc)?,Buffer::from(ctx,&qp)?,Buffer::from(ctx,&pm)?,Buffer::from(ctx,&pi)?,Buffer::from(ctx,&pc)?,Buffer::from(ctx,&pw)?,Buffer::from(ctx,&qo)?,Buffer::from(ctx,&qids)?,Buffer::from(ctx,&qcoef)?,Buffer::from(ctx,&po)?,Buffer::from(ctx,&pids)?,Buffer::from(ctx,&pcoef)?,Buffer::new(ctx,quad.len()*9*4)?,Buffer::new(ctx,penalties.len()*3*4)?,Buffer::new(ctx,rows*3*4)?,Buffer::from(ctx,&owned.iter().map(|v|*v as u32).collect::<Vec<_>>())?];
 // RHS, solution, residual, direction, preconditioned residual, diagonal,
 // mass, matrix direction and block reductions stay resident on the GPU.
 for values in [rows*3,rows*3,rows*3,rows*3,rows*3,rows,rows,rows*3,rows.max(quad.len()).max(penalties.len()).div_ceil(128)*3]{buffers.push(Buffer::new(ctx,values*4)?);}
 let nz=mu.len();let mut face_coeff=vec![0f32;rows*4];
 for (slot,&i)in d.needed.iter().enumerate(){for k in 0..nz{let n=slot*nz+k;
   for (side,other)in [(0,k.checked_sub(1)),(1,if k+1<nz{Some(k+1)}else{None})]{if let Some(j)=other{let face=k.max(j);let area=(RE+m.vertical.faces[face]).powi(2)*m.mesh.cells[i].area;let distance=(m.vertical.centers[k]-m.vertical.centers[j]).abs();face_coeff[4*n+side]=(area*(mu[k]+mu[j])*0.5*4./3./distance)as f32;face_coeff[4*n+2+side]=(area*(kappa[k]+kappa[j])*0.5/distance)as f32;}}
 }}
 buffers.push(Buffer::new(ctx,rows*4)?);buffers.push(Buffer::from(ctx,&face_coeff)?);
 Ok(Self{buffers,ctx,rows,quads:quad.len(),penalties:penalties.len(),nz})
 }
 pub fn solve(&mut self,b:&[Q],root:&[f64],diagonal:&[f64],dt:f64,thermal:bool,tolerance:f64)->Result<(Vec<Q>,usize,f64)>{
 ensure!(b.len()==self.rows && root.len()==self.rows && diagonal.len()==self.rows,"GPU CG sizes");let nc=if thermal{1}else{3};
 let scale=b.iter().zip(root).flat_map(|(q,r)|q[..nc].iter().map(move|x|(x/r).abs())).fold(0f64,f64::max);
 if scale==0.{return Ok((b.to_vec(),0,0.));}
 for ((out,q),r)in self.buffers[20].floats_mut(self.rows*3).chunks_exact_mut(3).zip(b).zip(root){for a in 0..3{out[a]=if a<nc{(q[a]/r/scale)as f32}else{0.};}}
 for (out,r)in self.buffers[26].floats_mut(self.rows).iter_mut().zip(root){*out=(r*r)as f32;}
 for (out,d)in self.buffers[25].floats_mut(self.rows).iter_mut().zip(diagonal){*out=*d as f32;}
 let mut pointers=self.buffers.iter().map(|b|b.0.as_ptr()).collect::<Vec<_>>();let mut iterations=0;let mut residual=0.;
 ensure!(unsafe{rns_transport_cg(self.ctx.as_ptr(),pointers.as_mut_ptr(),self.rows as u32,self.quads as u32,self.penalties as u32,thermal as u32,self.nz as u32,std::env::var("RNS_METAL_LINE_PRECONDITIONER").map(|v|v=="1").unwrap_or(false)as u32,dt as f32,tolerance as f32,&mut iterations,&mut residual)},"Resident Metal CG failed");
 let mut result=b.to_vec();for ((out,x),r)in result.iter_mut().zip(self.buffers[21].floats(self.rows*3).chunks_exact(3)).zip(root){for a in 0..nc{out[a]=x[a]as f64*scale*r;ensure!(out[a].is_finite(),"Nonfinite CG solution");}}
 Ok((result,iterations as usize,residual))
 }
 pub fn apply(&mut self,x:&[Q],root:&[f64],dt:f64,thermal:bool)->Result<Vec<Q>>{
 ensure!(x.len()==self.rows&&root.len()==self.rows,"Transport input size");let nc=if thermal{1}else{3};
 let scale=x.iter().zip(root).flat_map(|(q,r)|q[..nc].iter().map(move|x|(x/r).abs())).fold(0f64,f64::max);
 if scale==0.{return Ok(x.to_vec());}
 for ((out,q),r)in self.buffers[0].floats_mut(self.rows*3).chunks_exact_mut(3).zip(x).zip(root){for a in 0..3{out[a]=if a<nc{(q[a]/r/scale)as f32}else{0.};}}
 for (out,r)in self.buffers[1].floats_mut(self.rows).iter_mut().zip(root){*out=*r as f32;}
 let mut ptrs=self.buffers.iter().map(|b|b.0.as_ptr()).collect::<Vec<_>>();ensure!(unsafe{rns_transport_run(self.ctx.as_ptr(),ptrs.as_mut_ptr(),self.rows as u32,self.quads as u32,self.penalties as u32,thermal as u32,dt as f32)},"Transport GPU execution failed");
 let mut result=x.to_vec();for((out,gpu),r)in result.iter_mut().zip(self.buffers[18].floats(self.rows*3).chunks_exact(3)).zip(root){for a in 0..nc{out[a]+=dt*gpu[a]as f64*scale/r;ensure!(out[a].is_finite(),"Nonfinite GPU transport");}}Ok(result)
 }
}
// GPU mutation requires &mut self; immutable transport references in CPU
// gradient workers do not access these handles.
unsafe impl Sync for MetalTransport {}
