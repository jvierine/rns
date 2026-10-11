#!/bin/sh
# Quiet baseline and separate ERA5/solar adjustment tests, through 450 km.
set -eu
mode=${1:-quiet}
backend=${2:-cpu}
end=${3:-21600}
level=${4:-2}
case "$backend" in cpu|metal|auto) ;; *) exit 2;; esac
set --
case "$mode" in
 iau) set -- --era5 data/era5_20250306_23utc_t6_filtered.h5 --era5-iau-seconds 21600 ;;
 quiet) ;;
 thermodynamics) set -- --era5 data/era5_20250306_23utc.h5 --era5-hydrostatic-columns --era5-no-winds ;;
 winds) set -- --era5 data/era5_20250306_23utc.h5 --era5-hydrostatic-columns ;;
 solar) set -- --era5 data/era5_20250306_23utc.h5 --era5-hydrostatic-columns --era5-no-winds --solar-absorption 0.2 ;;
 *) echo 'Usage: run_control.sh [quiet|iau|thermodynamics|winds|solar] [cpu|metal|auto] [seconds] [mesh_level]' >&2; exit 2 ;;
esac
exec target/release/global-neutral-euler --level "$level" --layers 52 --case equilibrium \
 --background data/era5_msis_transport_450km_control.h5 \
 --rotation --sponge --discrete-hydrostatic --conservative-gravity-energy \
 --reconstruct-temperature --max-temperature-k 5000 \
 --transport data/era5_msis_transport_450km_control.h5 --transport-interval 5 \
 --end "$end" --cadence 600 --backend "$backend" "$@" \
 --output "output/control_${mode}_${backend}_level${level}_${end}s"
