#include <metal_stdlib>
using namespace metal;
kernel void transport_gradient(device const float*x [[buffer(0)]],device const float*root [[buffer(1)]],device const uint*meta [[buffer(2)]],device const uint*ids [[buffer(3)]],device const float*c [[buffer(4)]],device const float*params [[buffer(5)]],device float*s [[buffer(6)]],constant uint&count [[buffer(7)]],constant uint&thermal [[buffer(8)]],uint q [[thread_position_in_grid]]) {
 if(q>=count)return;uint center=meta[3*q],begin=meta[3*q+1],n=meta[3*q+2];float g[9]={0,0,0,0,0,0,0,0,0};uint nc=thermal?1:3;
 for(uint j=begin;j<begin+n;j++){uint neighbor=ids[j];for(uint a=0;a<nc;a++){float diff=x[3*neighbor+a]-x[3*center+a];for(uint b=0;b<3;b++)g[3*a+b]+=diff*c[3*j+b];}}
 float mu=params[3*q+1],kappa=params[3*q+2],trace=(g[0]+g[4]+g[8])/3;
 for(uint a=0;a<3;a++)for(uint b=0;b<3;b++)s[9*q+3*a+b]=thermal?(a==0?kappa*g[b]:0):mu*(g[3*a+b]+g[3*b+a]-(a==b?2*trace:0));
}
kernel void transport_penalty(device const float*x [[buffer(0)]],device const float*root [[buffer(1)]],device const uint*meta [[buffer(2)]],device const uint*ids [[buffer(3)]],device const float*c [[buffer(4)]],device const float*weights [[buffer(5)]],device float*out [[buffer(6)]],constant uint&count [[buffer(7)]],constant uint&thermal [[buffer(8)]],uint p [[thread_position_in_grid]]) {
 if(p>=count)return;uint begin=meta[2*p],n=meta[2*p+1],nc=thermal?1:3;float coefficient=weights[2*p+(thermal?1:0)];
 for(uint a=0;a<3;a++){float v=0;if(a<nc)for(uint j=begin;j<begin+n;j++)v+=c[j]*(x[3*ids[j]+a]-x[3*ids[begin]+a]);out[3*p+a]=coefficient*v;}
}
kernel void transport_gather(device const float*x [[buffer(0)]],device const float*root [[buffer(1)]],device const uint*offsets [[buffer(2)]],device const uint*ids [[buffer(3)]],device const float*c [[buffer(4)]],device const float*s [[buffer(5)]],device const uint*poffsets [[buffer(6)]],device const uint*pids [[buffer(7)]],device const float*pc [[buffer(8)]],device const float*pf [[buffer(9)]],device float*out [[buffer(10)]],device const uint*owned [[buffer(11)]],constant uint&count [[buffer(12)]],constant uint&thermal [[buffer(13)]],constant float&dt [[buffer(14)]],uint row [[thread_position_in_grid]]) {
 if(row>=count)return;uint nc=thermal?1:3;
 for(uint a=0;a<3;a++){float force=0;if(owned[row]&&a<nc){for(uint j=offsets[row];j<offsets[row+1];j++){uint q=ids[j];for(uint b=0;b<3;b++)force+=s[9*q+3*a+b]*c[3*j+b];}for(uint j=poffsets[row];j<poffsets[row+1];j++)force+=pc[j]*pf[3*pids[j]+a];}out[3*row+a]=force;}
}
kernel void cg_init(device const float*b [[buffer(0)]],device float*x [[buffer(1)]],device float*r [[buffer(2)]],device float*p [[buffer(3)]],device float*z [[buffer(4)]],device const float*d [[buffer(5)]],device const float*ap [[buffer(6)]],device const uint*owned [[buffer(7)]],constant uint&rows [[buffer(8)]],constant uint&thermal [[buffer(9)]],constant uint&mode [[buffer(10)]],uint n [[thread_position_in_grid]]){
 if(n>=rows)return;uint nc=thermal?1:3;for(uint a=0;a<3;a++){uint i=3*n+a;if(mode==1){x[i]=b[i];p[i]=b[i];}else if(mode==2){p[i]=x[i];}else{r[i]=(owned[n]&&a<nc)?b[i]-ap[i]:0;z[i]=r[i]/d[n];p[i]=z[i];}}
}
kernel void cg_identity(device const float*p [[buffer(0)]],device const float*f [[buffer(1)]],device const float*mass [[buffer(2)]],device float*ap [[buffer(3)]],device const uint*owned [[buffer(4)]],constant uint&rows [[buffer(5)]],constant uint&thermal [[buffer(6)]],constant float&dt [[buffer(7)]],uint n [[thread_position_in_grid]]){
 if(n>=rows)return;uint nc=thermal?1:3;for(uint a=0;a<3;a++){uint i=3*n+a;ap[i]=p[i]+((owned[n]&&a<nc)?dt*f[i]/mass[n]:0);}
}
kernel void cg_dot(device const float*b [[buffer(0)]],device const float*r [[buffer(1)]],device const float*p [[buffer(2)]],device const float*z [[buffer(3)]],device const float*mass [[buffer(4)]],device const float*ap [[buffer(5)]],device float*partial [[buffer(6)]],device const uint*owned [[buffer(7)]],constant uint&rows [[buffer(8)]],constant uint&thermal [[buffer(9)]],constant uint&mode [[buffer(10)]],uint n [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){
 threadgroup float s0[128],s1[128],s2[128];float v0=0,v1=0,v2=0;if(n<rows&&owned[n]){uint nc=thermal?1:3;for(uint a=0;a<nc;a++){uint i=3*n+a;if(mode==1)v0+=mass[n]*p[i]*ap[i];else{v0+=mass[n]*r[i]*z[i];v1+=mass[n]*r[i]*r[i];v2+=mass[n]*b[i]*b[i];}}}s0[lane]=v0;s1[lane]=v1;s2[lane]=v2;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=64;stride>0;stride/=2){if(lane<stride){s0[lane]+=s0[lane+stride];s1[lane]+=s1[lane+stride];s2[lane]+=s2[lane+stride];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lane==0){partial[3*group]=s0[0];partial[3*group+1]=s1[0];partial[3*group+2]=s2[0];}
}
kernel void cg_update(device float*x [[buffer(0)]],device float*r [[buffer(1)]],device const float*p [[buffer(2)]],device float*z [[buffer(3)]],device const float*d [[buffer(4)]],device const float*ap [[buffer(5)]],device const uint*owned [[buffer(6)]],constant uint&rows [[buffer(7)]],constant uint&thermal [[buffer(8)]],constant float&alpha [[buffer(9)]],uint n [[thread_position_in_grid]]){
 if(n>=rows||!owned[n])return;uint nc=thermal?1:3;for(uint a=0;a<nc;a++){uint i=3*n+a;x[i]+=alpha*p[i];r[i]-=alpha*ap[i];z[i]=r[i]/d[n];}
}
kernel void cg_direction(device float*p [[buffer(0)]],device const float*z [[buffer(1)]],device const uint*owned [[buffer(2)]],constant uint&rows [[buffer(3)]],constant uint&thermal [[buffer(4)]],constant float&beta [[buffer(5)]],uint n [[thread_position_in_grid]]){
 if(n>=rows||!owned[n])return;uint nc=thermal?1:3;for(uint a=0;a<nc;a++){uint i=3*n+a;p[i]=z[i]+beta*p[i];}
}
// Vertical line preconditioner; the full 3D operator remains unchanged.
kernel void cg_line(device const float*r [[buffer(0)]],device float*z [[buffer(1)]],device const float*face [[buffer(2)]],device const float*mass [[buffer(3)]],device float*cprime [[buffer(4)]],device const uint*owned [[buffer(5)]],constant uint&columns [[buffer(6)]],constant uint&nz [[buffer(7)]],constant uint&thermal [[buffer(8)]],constant float&dt [[buffer(9)]],uint col [[thread_position_in_grid]]){
 if(col>=columns)return;uint nc=thermal?1:3;uint base=col*nz;
 for(uint k=0;k<nz;k++){uint n=base+k;if(!owned[n]){cprime[n]=0;for(uint a=0;a<3;a++)z[3*n+a]=0;continue;}
  float left=-dt*face[4*n+(thermal?2:0)]/mass[n],right=-dt*face[4*n+(thermal?3:1)]/mass[n];float diagonal=1-left-right;
  float denominator=diagonal-(k>0?left*cprime[n-1]:0);cprime[n]=right/denominator;
  for(uint a=0;a<nc;a++)z[3*n+a]=(r[3*n+a]-(k>0?left*z[3*(n-1)+a]:0))/denominator;
 }
 for(uint k=nz-1;k>0;k--){uint n=base+k-1;for(uint a=0;a<nc;a++)z[3*n+a]-=cprime[n]*z[3*(n+1)+a];}
}
kernel void cg_energy(device const float*p [[buffer(0)]],device const float*mass [[buffer(1)]],device const uint*owned [[buffer(2)]],device const float*qp [[buffer(3)]],device const float*s [[buffer(4)]],device const float*pw [[buffer(5)]],device const float*pf [[buffer(6)]],device float*partial [[buffer(7)]],constant uint&rows [[buffer(8)]],constant uint&quads [[buffer(9)]],constant uint&penalties [[buffer(10)]],constant uint&thermal [[buffer(11)]],constant float&dt [[buffer(12)]],uint n [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){
 threadgroup float energy[128];float e=0;uint nc=thermal?1:3;
 if(n<rows&&owned[n])for(uint a=0;a<nc;a++)e+=mass[n]*p[3*n+a]*p[3*n+a];
 if(n<quads){float norm=0;for(uint a=0;a<(thermal?3:9);a++)norm+=s[9*n+a]*s[9*n+a];e+=dt*qp[3*n]*norm/(thermal?qp[3*n+2]:2*qp[3*n+1]);}
 if(n<penalties){float norm=0;for(uint a=0;a<nc;a++)norm+=pf[3*n+a]*pf[3*n+a];e+=dt*norm/pw[2*n+(thermal?1:0)];}
 energy[lane]=e;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint stride=64;stride>0;stride/=2){if(lane<stride)energy[lane]+=energy[lane+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}if(lane==0){partial[3*group]=energy[0];partial[3*group+1]=0;partial[3*group+2]=0;}
}
