#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
export HDF5_DIR="${HDF5_DIR:-/opt/anaconda3}"
cargo build --release --no-default-features --bin benchmark_metal
xcrun swiftc -O benchmarks/apple/run.swift -o /tmp/rns-native-metal
target/release/benchmark_metal "${1:-1000000}" | tee benchmarks/apple/cpu_timing.txt
/tmp/rns-native-metal benchmarks/apple/input.bin benchmarks/apple/hlle.metal benchmarks/apple/output.bin | tee benchmarks/apple/metal_timing.txt
/tmp/rns-native-metal benchmarks/apple/input.bin benchmarks/apple/hlle_soa.metal benchmarks/apple/output_soa.bin | tee benchmarks/apple/metal_soa_timing.txt
conda run -n base python benchmarks/apple/analyze.py
