# Apple GPU benchmark

Run on macOS:

```sh
benchmarks/apple/run.sh
```

One million seven-component HLLE face fluxes on an Apple M4 Pro:

| Implementation | Mean time |
|---|---:|
| Rust CPU, f64, one thread | 26.14 ms |
| Rust CPU, f64, 14 threads | 6.16 ms |
| Rust CPU, normalized f32, 14 threads | 3.14 ms |
| Native Metal, f32, 256 threads per group | 0.95 ms |

Metal is 6.5× faster than the existing parallel f64 flux calculation, or 3.3× faster than the normalized f32 CPU calculation. The structure-of-arrays variant did not improve this run.

These are flux-kernel timings, not complete simulation timings. Inputs are prepared outside the timed loop. Metal wall time includes command submission and completion; buffers are resident in shared memory. CPU parallel timing includes thread creation. Both implementations write all seven outputs. Compilation is excluded.

The test contains weak perturbations with CH₄ and O₂. Metal's maximum relative field L2 difference from the f64 reference is 3.51e-5. Timings, errors, and source hashes are in `cpu_benchmark.h5`. The Metal kernels use precise arithmetic; fast math is disabled.

`hlle.metal` and `hlle_soa.metal` are hand-written Metal; `run.swift` is the native dispatch harness. The Rust reference is `src/bin/benchmark_metal.rs`.
