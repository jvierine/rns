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
 out[i*7+f]=(sr*fl-sl*fr+sl*sr*(a[b+7+f]-a[b+f]))/(sr-sl);
 }
}
