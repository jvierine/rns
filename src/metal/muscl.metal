#include <metal_stdlib>
using namespace metal;
kernel void slopes(device const float*p [[buffer(0)]],device const uint*topo [[buffer(1)]],device const float*g [[buffer(2)]],device const float*z [[buffer(3)]],device float*out [[buffer(4)]],constant uint&nz [[buffer(5)]],constant uint&cells [[buffer(6)]],uint idx [[thread_position_in_grid]]) {
 if(idx>=cells*7)return;uint cell=idx/7,f=idx%7,k=cell%nz,col=cell/nz;
 for(uint axis=0;axis<4;axis++)out[cell*28+f*4+axis]=0;
 if(!topo[col*4+3])return;
 float center=p[cell*18+f],lo=center,hi=center;float3 grad=0;
 for(uint j=0;j<3;j++){float v=p[(topo[col*4+j]*nz+k)*18+f];lo=min(lo,v);hi=max(hi,v);grad+=float3(g[col*18+j*3],g[col*18+j*3+1],g[col*18+j*3+2])*(v-center);}
 float theta=1;
 for(uint j=0;j<3;j++){float change=dot(grad,float3(g[col*18+9+j*3],g[col*18+10+j*3],g[col*18+11+j*3]));if(change>0)theta=min(theta,(hi-center)/change);if(change<0)theta=min(theta,(lo-center)/change);}
 grad*=clamp(theta,0.f,1.f);for(uint a=0;a<3;a++)out[cell*28+f*4+a]=grad[a];
 if(k>0 && k+1<nz){float dl=z[k*3],dr=z[k*3+1],w=z[k*3+2];float vertical_center=p[cell*18+9+f];float below=p[(cell-1)*18+9+f],above=p[(cell+1)*18+9+f];
 if(f==4){bool thermal=(uint(p[cell*18+16])&2)!=0;uint nb[2]={cell-1,cell+1};float v[2]={below,above};for(uint j=0;j<2;j++){if(((uint(p[nb[j]*18+16])&2)!=0)!=thermal){float dp=p[nb[j]*18+17],dr=p[nb[j]*18+9];v[j]=thermal?(dp-dr)/(1+dr):dp;}}below=v[0];above=v[1];}
 float l=(vertical_center-below)/dl,r=(above-vertical_center)/dr;
 if(l*r>0)out[cell*28+f*4+3]=(l>0?1.f:-1.f)*min(2*abs(l),min(2*abs(r),abs((dl*l+dr*r)/(dl+dr))))*w;}
}
kernel void faces(device const float*p [[buffer(0)]],device const float*s [[buffer(1)]],device const uint*ids [[buffer(2)]],device const float*g [[buffer(3)]],device float*out [[buffer(4)]],constant uint&count [[buffer(5)]],uint face [[thread_position_in_grid]]) {
 if(face>=count)return;uint b=face*18,o=face*24;float3 n=float3(g[b],g[b+1],g[b+2]);
 for(uint side=0;side<2;side++){uint cell=ids[face*5+side];uint shift=ids[face*5+4]?9:0;float v[7];float norm=0;
 for(uint f=0;f<7;f++){v[f]=cell==UINT_MAX?0:p[cell*18+shift+f];norm+=abs(v[f]);}
 if(cell!=UINT_MAX && (uint(p[cell*18+shift+7])&1)==0 && g[b+17]>0){
 for(uint f=0;f<7;f++){uint j=cell*28+f*4;v[f]+=s[j]*g[b+3+side*3]+s[j+1]*g[b+4+side*3]+s[j+2]*g[b+5+side*3]+s[j+3]*g[b+9+side];}
 if(v[0]<=-1 || v[4]<=-1 || v[5]<0 || v[6]<0 || v[5]+v[6]>=1)for(uint f=0;f<7;f++)v[f]=p[cell*18+shift+f];}
 if(cell!=UINT_MAX && (uint(p[cell*18+shift+7])&2)!=0){
 // Retain the f64-packed centre pressure perturbation. Recomputing it as
 // (1+dT)*(1+drho)-1 loses weak pressure signals through f32 cancellation.
 float dr=v[0]-p[cell*18+shift],dt=v[4]-p[cell*18+shift+4];
 v[4]=p[cell*18+shift+8]+(1+p[cell*18+shift+4])*dr+(1+p[cell*18+shift])*dt+dr*dt;
 }
 if(ids[face*5+2+side]){float3 u=float3(v[1],v[2],v[3]);u-=2*dot(u,n)*n;v[1]=u.x;v[2]=u.y;v[3]=u.z;}
 float rho=1+v[0],ch=rho*v[5],ox=rho*v[6];float dg=v[0]+ch*(g[b+12]-1)+ox*(g[b+13]-1),dc=v[0]+ch*(g[b+14]-1)+ox*(g[b+15]-1);float factor=g[b+16],gm=g[b+11]-1;
 out[o+side*7]=v[0];for(uint a=0;a<3;a++)out[o+side*7+a+1]=rho*v[a+1]*factor;
 float kin=.5*rho*factor*factor*g[b+11]*gm*(v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
 out[o+side*7+4]=(v[4]+dc+v[4]*dc-dg)/(1+dg)+kin;out[o+22+side]=v[4];out[o+side*7+5]=ch;out[o+side*7+6]=ox;
 }
 for(uint a=0;a<3;a++)out[o+14+a]=g[b+a];for(uint f=0;f<5;f++)out[o+17+f]=g[b+11+f];
}
#include <metal_stdlib>
using namespace metal;
kernel void hlle(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],constant uint &count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=count)return;
 const uint b=i*24;
 const float gamma=a[b+17],gm=gamma-1,rl=1+a[b],rr=1+a[b+7];
 const float gl=a[b]+a[b+5]*(a[b+18]-1)+a[b+6]*(a[b+19]-1);
 const float gr=a[b+7]+a[b+12]*(a[b+18]-1)+a[b+13]*(a[b+19]-1);
 const float vl=a[b]+a[b+5]*(a[b+20]-1)+a[b+6]*(a[b+21]-1);
 const float vr=a[b+7]+a[b+12]*(a[b+20]-1)+a[b+13]*(a[b+21]-1);
 const float pl=a[b+22],pr=a[b+23];
 const float ul=(a[b+1]*a[b+14]+a[b+2]*a[b+15]+a[b+3]*a[b+16])/rl;
 const float ur=(a[b+8]*a[b+14]+a[b+9]*a[b+15]+a[b+10]*a[b+16])/rr;
 const float cl=sqrt((1+gm*(1+gl)/(1+vl))*(1+pl)/(gamma*rl));
 const float cr=sqrt((1+gm*(1+gr)/(1+vr))*(1+pr)/(gamma*rr));
 const float sl=min(0.f,min(ul-cl,ur-cr)),sr=max(0.f,max(ul+cl,ur+cr));
 #pragma unroll
 for(uint f=0;f<7;f++) {
 float fl=a[b+f]*ul,fr=a[b+7+f]*ur;
 if(f==0){fl=rl*ul;fr=rr*ur;}
 if(f>0 && f<4){fl+=pl*a[b+13+f]/gamma;fr+=pr*a[b+13+f]/gamma;}
 if(f==4){fl=(1+a[b+4]+gm*(1+pl))*ul;fr=(1+a[b+11]+gm*(1+pr))*ur;}
 // Equivalent HLLE tracer flux, with nonnegative one-sided coefficients.
 // Avoid cancellation between diffusion and advection at a zero-species front.
 if(f>=5)out[i*7+f]=(sr*(ul-sl)*a[b+f]+sl*(sr-ur)*a[b+7+f])/(sr-sl);
 else out[i*7+f]=(sr*fl-sl*fr+sl*sr*(a[b+7+f]-a[b+f]))/(sr-sl);
 }
}
