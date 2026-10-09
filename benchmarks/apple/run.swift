import Foundation
import Metal
let args=CommandLine.arguments
let raw=try Data(contentsOf:URL(fileURLWithPath:args[1]))
let n=raw.count/(24*4)
let soa=args[2].contains("soa")
var data=raw
if soa {var values=[Float](repeating:0,count:n*24);raw.withUnsafeBytes {ptr in let a=ptr.bindMemory(to:Float.self);for f in 0..<24 {for i in 0..<n {values[f*n+i]=a[i*24+f]}}};data=values.withUnsafeBytes {Data($0)}}
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {fatalError("No Metal GPU")}
let source=try String(contentsOfFile:args[2],encoding:.utf8)
let options=MTLCompileOptions();options.fastMathEnabled=false
let lib=try device.makeLibrary(source:source,options:options)
let pipeline=try device.makeComputePipelineState(function:lib.makeFunction(name:soa ? "hlle_soa" : "hlle")!)
let input=data.withUnsafeBytes {device.makeBuffer(bytes:$0.baseAddress!,length:data.count,options:.storageModeShared)!}
let output=device.makeBuffer(length:n*7*4,options:.storageModeShared)!
var count=UInt32(n)
func execute(_ threads:Int)->(Double,Double) {
 let start=DispatchTime.now().uptimeNanoseconds
 let cb=queue.makeCommandBuffer()!;let e=cb.makeComputeCommandEncoder()!
 e.setComputePipelineState(pipeline);e.setBuffer(input,offset:0,index:0);e.setBuffer(output,offset:0,index:1);e.setBytes(&count,length:4,index:2)
 e.dispatchThreads(MTLSize(width:n,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1));e.endEncoding();cb.commit();cb.waitUntilCompleted()
 if let error=cb.error {fatalError("\(error)")}
 return (Double(DispatchTime.now().uptimeNanoseconds-start)/1e9,cb.gpuEndTime-cb.gpuStartTime)
}
print("device \(device.name), faces \(n)")
for threads in [64,128,256] {
 _=execute(threads);var wall:[Double]=[];var gpu:[Double]=[]
 for _ in 0..<20 {let t=execute(threads);wall.append(t.0);gpu.append(t.1)}
 let avg=wall.reduce(0,+)/Double(wall.count);let gav=gpu.reduce(0,+)/Double(gpu.count)
 print("threads \(threads) wall \(avg) gpu \(gav)")
}
var result=Data(bytes:output.contents(),count:output.length)
if soa {var values=[Float](repeating:0,count:n*7);let a=output.contents().bindMemory(to:Float.self,capacity:n*7);for f in 0..<7 {for i in 0..<n {values[i*7+f]=a[f*n+i]}};result=values.withUnsafeBytes{Data($0)}}
try result.write(to:URL(fileURLWithPath:args[3]))
