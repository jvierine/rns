#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
extern "C" {
void* rns_metal_new(const char* source) { @autoreleasepool {
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();if(!d)return nullptr;
 MTLCompileOptions* opts=[MTLCompileOptions new];opts.fastMathEnabled=NO;
 NSError* err=nil;id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&err];
 if(!lib){fprintf(stderr,"Metal compile: %s\n",err.description.UTF8String);return nullptr;}
 NSMutableArray* p=[NSMutableArray new];
 for(NSString* name in @[@"slopes",@"faces",@"hlle"]) {id<MTLComputePipelineState> s=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];if(!s)return nullptr;[p addObject:s];}
 NSDictionary* context=@{@"device":d,@"queue":[d newCommandQueue],@"pipelines":p};
 return (void*)CFBridgingRetain(context);
}}
void rns_metal_release(void* ptr){if(ptr)CFBridgingRelease(ptr);}
void* rns_metal_buffer(void* ctx,size_t bytes) { @autoreleasepool {id<MTLDevice> d=((__bridge NSDictionary*)ctx)[@"device"];id<MTLBuffer> b=[d newBufferWithLength:bytes options:MTLResourceStorageModeShared];return b?(void*)CFBridgingRetain(b):nullptr;}}
void* rns_metal_contents(void* ptr){return [(__bridge id<MTLBuffer>)ptr contents];}
bool rns_metal_run(void* ctx,void** ptr,uint32_t cells,uint32_t nz,uint32_t nfaces) { @autoreleasepool {
 NSDictionary* c=(__bridge NSDictionary*)ctx;id<MTLCommandQueue> queue=c[@"queue"];NSArray* pipelines=c[@"pipelines"];
 id<MTLCommandBuffer> cmd=[queue commandBuffer];
 // primitive, topology, column geometry, vertical geometry, slopes, face ids, face geometry, packed faces, output
 const int bindings[3][6]={{0,1,2,3,4,-1},{0,4,5,6,7,-1},{7,8,-1,-1,-1,-1}};
 const uint32_t counts[3]={cells*7,nfaces,nfaces};
 for(int k=0;k<3;k++) {id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];[e setComputePipelineState:pipelines[k]];
 int b=0;for(;b<6 && bindings[k][b]>=0;b++)[e setBuffer:(__bridge id<MTLBuffer>)ptr[bindings[k][b]] offset:0 atIndex:b];
 if(k==0){[e setBytes:&nz length:4 atIndex:b++];[e setBytes:&cells length:4 atIndex:b];}else [e setBytes:&nfaces length:4 atIndex:b];
 [e dispatchThreads:MTLSizeMake(counts[k],1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
 [cmd commit];[cmd waitUntilCompleted];
 if(cmd.status==MTLCommandBufferStatusError){fprintf(stderr,"Metal execution: %s\n",cmd.error.description.UTF8String);return false;}return true;
}}
bool rns_metal_flux(void* ctx,void* input,void* output,uint32_t count) { @autoreleasepool {
 NSDictionary*c=(__bridge NSDictionary*)ctx;id<MTLCommandQueue>q=c[@"queue"];id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][2]];[e setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:1];[e setBytes:&count length:4 atIndex:2];[e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[cmd commit];[cmd waitUntilCompleted];return cmd.status!=MTLCommandBufferStatusError;
}}
}
extern "C" {
void* rns_transport_new(const char* source){@autoreleasepool{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return nullptr;MTLCompileOptions*opts=[MTLCompileOptions new];opts.fastMathEnabled=NO;NSError*err=nil;
 id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&err];if(!lib){fprintf(stderr,"Transport Metal compile: %s\n",err.description.UTF8String);return nullptr;}
 NSMutableArray*p=[NSMutableArray new];for(NSString*name in @[@"transport_gradient",@"transport_penalty",@"transport_gather",@"cg_init",@"cg_identity",@"cg_dot",@"cg_update",@"cg_direction",@"cg_line",@"cg_energy"]){id<MTLComputePipelineState>s=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:name]error:&err];if(!s)return nullptr;[p addObject:s];}
 return(void*)CFBridgingRetain(@{@"device":d,@"queue":[d newCommandQueue],@"pipelines":p});
}}
bool rns_transport_run(void*ctx,void**ptr,uint32_t rows,uint32_t quads,uint32_t penalties,uint32_t thermal,float dt){@autoreleasepool{
 NSDictionary*c=(__bridge NSDictionary*)ctx;id<MTLCommandBuffer>cmd=[c[@"queue"] commandBuffer];
 const int binding[3][13]={{0,1,2,3,4,5,16,-1},{0,1,6,7,8,9,17,-1},{0,1,10,11,12,16,13,14,15,17,18,19,-1}};
 const uint32_t counts[3]={quads,penalties,rows};
 for(int k=0;k<3;k++){if(!counts[k])continue;id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][k]];int b=0;for(;binding[k][b]>=0;b++)[e setBuffer:(__bridge id<MTLBuffer>)ptr[binding[k][b]]offset:0 atIndex:b];[e setBytes:&counts[k]length:4 atIndex:b++];[e setBytes:&thermal length:4 atIndex:b++];if(k==2)[e setBytes:&dt length:4 atIndex:b];[e dispatchThreads:MTLSizeMake(counts[k],1,1)threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
 [cmd commit];[cmd waitUntilCompleted];if(cmd.status==MTLCommandBufferStatusError){fprintf(stderr,"Transport Metal: %s\n",cmd.error.description.UTF8String);return false;}return true;
}}
}
#include <initializer_list>
#include <algorithm>

#include <cmath>
extern "C" bool rns_transport_cg(void*ctx,void**ptr,uint32_t rows,uint32_t quads,uint32_t penalties,uint32_t thermal,uint32_t nz,uint32_t line_enabled,float dt,float tolerance,uint32_t*iterations,double*residual){@autoreleasepool{
 NSDictionary*c=(__bridge NSDictionary*)ctx;id<MTLCommandQueue>queue=c[@"queue"];uint32_t groups=(rows+127)/128;uint32_t sum_groups=groups;ptr[0]=ptr[23];
 auto encode=[&](id<MTLCommandBuffer>cmd,int pipeline,std::initializer_list<int>bindings,uint32_t count,uint32_t mode,float value,bool integer,bool dot){
  if(!count)return;id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][pipeline]];int b=0;for(int i:bindings){[e setBuffer:(__bridge id<MTLBuffer>)ptr[i]offset:0 atIndex:b++];}[e setBytes:&count length:4 atIndex:b++];[e setBytes:&thermal length:4 atIndex:b++];if(pipeline>=3){if(integer)[e setBytes:&mode length:4 atIndex:b];else[e setBytes:&value length:4 atIndex:b];}
  uint32_t threads=dot?groups*128:count;[e dispatchThreads:MTLSizeMake(threads,1,1)threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
 };
 auto matrix=[&](id<MTLCommandBuffer>cmd){
  encode(cmd,0,{0,1,2,3,4,5,16},quads,0,0,false,false);
  encode(cmd,1,{0,1,6,7,8,9,17},penalties,0,0,false,false);
  // Gather has one additional float argument after count and thermal.
  id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][2]];int b=0;for(int i:{0,1,10,11,12,16,13,14,15,17,18,19})[e setBuffer:(__bridge id<MTLBuffer>)ptr[i]offset:0 atIndex:b++];[e setBytes:&rows length:4 atIndex:b++];[e setBytes:&thermal length:4 atIndex:b++];[e setBytes:&dt length:4 atIndex:b];[e dispatchThreads:MTLSizeMake(rows,1,1)threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
  encode(cmd,4,{23,18,26,27,19},rows,0,dt,false,false);
 };
 auto line=[&](id<MTLCommandBuffer>cmd){if(!line_enabled)return;uint32_t columns=rows/nz;id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][8]];int b=0;for(int i:{22,24,30,26,29,19})[e setBuffer:(__bridge id<MTLBuffer>)ptr[i]offset:0 atIndex:b++];[e setBytes:&columns length:4 atIndex:b++];[e setBytes:&nz length:4 atIndex:b++];[e setBytes:&thermal length:4 atIndex:b++];[e setBytes:&dt length:4 atIndex:b];[e dispatchThreads:MTLSizeMake(columns,1,1)threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];};
 auto init=[&](id<MTLCommandBuffer>cmd,uint32_t mode){encode(cmd,3,{20,21,22,23,24,25,27,19},rows,mode,0,true,false);};
 auto dot=[&](id<MTLCommandBuffer>cmd,uint32_t mode){
   if(mode==0){sum_groups=groups;encode(cmd,5,{20,22,23,24,26,27,28,19},rows,mode,0,true,true);return;}
   uint32_t count=std::max(rows,std::max(quads,penalties));sum_groups=(count+127)/128;
   id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:c[@"pipelines"][9]];int b=0;for(int i:{23,26,19,5,16,9,17,28})[e setBuffer:(__bridge id<MTLBuffer>)ptr[i]offset:0 atIndex:b++];for(uint32_t value:{rows,quads,penalties,thermal})[e setBytes:&value length:4 atIndex:b++];[e setBytes:&dt length:4 atIndex:b];[e dispatchThreads:MTLSizeMake(sum_groups*128,1,1)threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
 };
 auto complete=[&](id<MTLCommandBuffer>cmd){[cmd commit];[cmd waitUntilCompleted];return cmd.status!=MTLCommandBufferStatusError;};
 auto sums=[&](double*out){float*v=(float*)[(__bridge id<MTLBuffer>)ptr[28]contents];out[0]=out[1]=out[2]=0;for(uint32_t i=0;i<sum_groups;i++)for(int a=0;a<3;a++)out[a]+=v[3*i+a];};
 id<MTLCommandBuffer>cmd=[queue commandBuffer];init(cmd,1);matrix(cmd);init(cmd,0);line(cmd);encode(cmd,7,{23,24,19},rows,0,0,false,false);dot(cmd,0);if(!complete(cmd))return false;double values[3];sums(values);double rz=values[0],rr=values[1],bb=values[2];*iterations=0;
 // Recursive float32 residuals can drift far below b-Ax for the stratified
 // atmosphere. Recompute the actual residual and restart the direction whenever
 // convergence is reported, and periodically before conjugacy is lost.
 while(bb>0&&*iterations<1200){
  if(std::sqrt(rr/bb)<=tolerance || (*iterations>0 && *iterations%128==0)){
   cmd=[queue commandBuffer];init(cmd,2);matrix(cmd);init(cmd,0);line(cmd);dot(cmd,0);
   if(!complete(cmd))return false;sums(values);rz=values[0];rr=values[1];
   if(std::sqrt(rr/bb)<=tolerance)break;
   cmd=[queue commandBuffer];encode(cmd,7,{23,24,19},rows,0,0,false,false);
   if(!complete(cmd))return false;
  }
  cmd=[queue commandBuffer];matrix(cmd);dot(cmd,1);if(!complete(cmd))return false;double pa[3];sums(pa);if(!(pa[0]>0&&std::isfinite(pa[0]))){fprintf(stderr,"CG denominator failure pa=%g iter=%u\n",pa[0],*iterations);return false;}float alpha=(float)(rz/pa[0]);
  cmd=[queue commandBuffer];encode(cmd,6,{21,22,23,24,25,27,19},rows,0,alpha,false,false);line(cmd);dot(cmd,0);if(!complete(cmd))return false;sums(values);double next=values[0];rr=values[1];float beta=(float)(next/rz);rz=next;
  cmd=[queue commandBuffer];encode(cmd,7,{23,24,19},rows,0,beta,false,false);if(!complete(cmd))return false;(*iterations)++;
 }
 cmd=[queue commandBuffer];init(cmd,2);matrix(cmd);init(cmd,0);dot(cmd,0);if(!complete(cmd))return false;sums(values);*residual=bb>0?std::sqrt(values[1]/bb):0;if(!std::isfinite(*residual)||*residual>=0.1)fprintf(stderr,"CG residual failure r=%g iter=%u\n",*residual,*iterations);return std::isfinite(*residual)&&*residual<0.1;
}}
