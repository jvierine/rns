#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release --bin global-neutral-euler
case "${1:-lamb}" in
 lamb) mpirun -n "${RANKS:-48}" target/release/global-neutral-euler --level 5 --layers 12 --dz 5000 --end 60000 --cadence 600 --sigma-km 500 --case lamb --sponge --output output/tonga_lamb ;;
 msis) mpirun -n "${RANKS:-48}" target/release/global-neutral-euler --level 5 --layers 80 --dz 2500 --end 60000 --cadence 1200 --sigma-km 250 --case heat --background data/tonga_msis_200km.h5 --sponge --output output/tonga_msis ;;
 *) echo 'Usage: run.sh lamb|msis'; exit 2 ;;
esac
