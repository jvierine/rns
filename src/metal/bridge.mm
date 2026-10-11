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
