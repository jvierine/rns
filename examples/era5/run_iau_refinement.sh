#!/bin/bash
# Same IAU experiment at doubled horizontal resolution; CPU reference first.
set -euo pipefail
level=${1:-3}
end=${2:-43200}
tag="era5_iau_level${level}_${end}s"
mkdir -p output
mkdir "output/${tag}.lock"
trap 'rmdir "output/${tag}.lock"' EXIT
solver=${RNS_SOLVER:-target/release/global-neutral-euler}
base=("$solver" --level "$level" --layers 52 --case equilibrium
 --background data/era5_msis_transport_450km_control.h5
 --era5 data/era5_20250306_23utc_t6_filtered.h5 --era5-iau-seconds 21600
 --rotation --sponge --discrete-hydrostatic --conservative-gravity-energy
 --reconstruct-temperature --max-temperature-k 5000
 --transport data/era5_msis_transport_450km_control.h5 --transport-interval 5)
mkdir "output/${tag}_sources"
shasum -a 256 "$solver" src/{main,lib,initialization,era5,solver,transport,native_metal,physics}.rs src/metal/muscl.metal data/era5_msis_transport_450km_control.h5 data/era5_20250306_23utc_t6_filtered.h5 > "output/${tag}_sources/manifest.sha256"
cp "$solver" "output/${tag}_sources/solver"
base[0]="output/${tag}_sources/solver"
for backend in cpu metal; do
 out="output/${tag}_${backend}600"
 test ! -e "$out"
 RNS_REQUIRE_GPU=1 "${base[@]}" --backend "$backend" --end 600 --cadence 60 --output "$out" > "$out.log" 2>&1
 test -f "$out/run.closed"
 echo "CLOSED $out"
done
conda run -n base python scripts/validate_era5_backends.py "output/${tag}_cpu600" "output/${tag}_metal600" --end 600 --output "data/${tag}_cpu_metal_validation.h5"
out="output/${tag}_metal"
test ! -e "$out"
RNS_REQUIRE_GPU=1 "${base[@]}" --backend metal --end "$end" --cadence 600 --output "$out" > "$out.log" 2>&1
test -f "$out/run.closed"
shasum -a 256 -c "output/${tag}_sources/manifest.sha256"
conda run -n base python scripts/validate_era5_iau.py "$out" --output "data/${tag}_validation.h5"
conda run -n base python scripts/validate_molecular_run.py "$out" --input data/era5_msis_transport_450km_control.h5 --output "data/${tag}_molecular_validation.h5"
echo "VALIDATED $out"
