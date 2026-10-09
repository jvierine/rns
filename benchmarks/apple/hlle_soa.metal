#include <metal_stdlib>
using namespace metal;
kernel void hlle_soa(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],constant uint &count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=count)return;
 const uint b=i*24;
 const float gamma=a[17*count+i],gm=gamma-1,rl=1+a[i],rr=1+a[7*count+i];
 const float gl=a[i]+a[5*count+i]*(a[18*count+i]-1)+a[6*count+i]*(a[19*count+i]-1);
 const float gr=a[7*count+i]+a[12*count+i]*(a[18*count+i]-1)+a[13*count+i]*(a[19*count+i]-1);
 const float vl=a[i]+a[5*count+i]*(a[20*count+i]-1)+a[6*count+i]*(a[21*count+i]-1);
 const float vr=a[7*count+i]+a[12*count+i]*(a[20*count+i]-1)+a[13*count+i]*(a[21*count+i]-1);
 const float pl=a[22*count+i],pr=a[23*count+i];
 const float ul=(a[1*count+i]*a[14*count+i]+a[2*count+i]*a[15*count+i]+a[3*count+i]*a[16*count+i])/rl;
 const float ur=(a[8*count+i]*a[14*count+i]+a[9*count+i]*a[15*count+i]+a[10*count+i]*a[16*count+i])/rr;
 const float cl=sqrt((1+gm*(1+gl)/(1+vl))*(1+pl)/(gamma*rl));
 const float cr=sqrt((1+gm*(1+gr)/(1+vr))*(1+pr)/(gamma*rr));
 const float sl=min(0.f,min(ul-cl,ur-cr)),sr=max(0.f,max(ul+cl,ur+cr));
 #pragma unroll
 for(uint f=0;f<7;f++) {
 float fl=a[f*count+i]*ul,fr=a[(7+f)*count+i]*ur;
 if(f==0){fl=rl*ul;fr=rr*ur;}
 if(f>0 && f<4){fl+=pl*a[(13+f)*count+i]/gamma;fr+=pr*a[(13+f)*count+i]/gamma;}
 if(f==4){fl=(1+a[4*count+i]+gm*(1+pl))*ul;fr=(1+a[11*count+i]+gm*(1+pr))*ur;}
 out[f*count+i]=(sr*fl-sl*fr+sl*sr*(a[(7+f)*count+i]-a[f*count+i]))/(sr-sl);
 }
}
