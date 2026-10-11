# ERA5 initialization tests

A hydrostatic column does not establish horizontal dynamical balance. Interpolation from ERA5 pressure levels onto a flat, coarse icosahedral grid can change the relation between wind, pressure gradients and rotation. Below-terrain ERA5 values are also retained in this flat-ground example. Neither issue is fixed by a vertical hydrostatic projection alone.

## Spatial projection trial

`scripts/filter_era5_initial.py` projects each initial altitude onto real spherical harmonics through degree 6, with multiplier

    exp[-l(l+1) (1500 km / Earth radius)^2 / 2].

At the level-2 diagnostic grid (320 triangles), this retains planetary scales rather than interpolating finer ERA5 structures directly onto individual cells. The degree-6 nominal wavelength is about 6200 km. This is a trial cutoff, not an optimized atmospheric resolution.

Pressure and temperature are filtered in logarithmic coordinates, then rescaled to retain their original area means at each input altitude. Wind is filtered in Cartesian Earth-fixed components, then projected onto the tangent plane. Initial vertical wind is zero. Density is recomputed from pressure and temperature. Hydrostatic columns are subsequently rebuilt by the solver. Positivity, constant preservation, retained-harmonic response and wind tangency are checked by the preparation script. The input records the filter, coefficients, source hash and script hash. Changing this initial state can change total initial mass and energy; this is not a conservative time-evolution operation.

The six-hour test uses the frozen v3 solver, identical 450-km background, rotation, sponge and molecular transport; solar heating is off. Rust CPU and native-Metal 600-s controls precede the Metal six-hour test. Original ERA5 input and previous controls remain unchanged. No filter acts on the subsequent solution or future rocket sources.

## Dynamical initialization

Spatial filtering removes unresolved scales, but cannot guarantee balance of the retained winds and pressure gradients. Digital-filter initialization filters a short model-generated time sequence to reduce artificial fast adjustment modes (Lynch and Huang, 1992, *Monthly Weather Review*, https://maths.ucd.ie/~plynch/Publications/adfi.pdf). It is distinct from spatial smoothing. A future implementation must use a separate initialization period with rocket forcing disabled, preserve the desired slow circulation, check mass/energy changes, and then start the scientific simulation with filtering disabled. Backward integration of the current dissipative transport is not a valid shortcut.

Current evidence and comparisons are written under `output/era5_isolation_v1` and `data/era5_isolation_*`. Filtering is not accepted as a production fix merely because a run completes; the wind/pressure adjustment, grid-scale structure and physical wave propagation must be compared.

## Isolation results (six-hour, unfiltered controls)

The unforced reference atmosphere remained exactly at rest on Metal for six hours; the Rust CPU 600-s reference also retained zero velocity. Relative pressure roundoff was 2.22e-16. ERA5 thermodynamics without winds or solar reached an all-time maximum upper vertical speed of 38.67 m/s. Adding the imported winds with solar still disabled increased that peak to 469.88 m/s at 412.5 km after 1200 s. Thus the large startup transient is associated primarily with the imported wind/thermodynamic combination in this coarse model, rather than the continuous solar source. This does not by itself distinguish under-resolution from dynamical imbalance or a flow-dependent discretization defect; the filtered CPU/Metal controls are a separate test.

The initial-state T6 projection retains pressure and temperature area means to roundoff; its maximum area-mean density change is 0.1716%, and its maximum horizontal input wind drops from 127.58 to 53.48 m/s. This changes the initial circulation; it is not an added damping term in the evolution. Input checks are archived in `data/era5_t6_input_filter_validation.h5`. CPU/Metal 600-s evolution agrees with maximum relative L2 2.55477e-6 (`data/era5_filter_t6_cpu_metal_validation.h5`).

The solar-only uniform-reference control uses the solver's default epoch rather than ERA5's date; it is an isolated mechanism test, not a matched March-2025 solar forecast. Its all-time upper vertical speed was 7.13 m/s. All four individual controls closed successfully and were validated externally. The original campaign shell emitted a post-run path-invocation error after its final solver closed because its script was edited while the loop was running; none of the solver outputs was restarted or altered. The script has been restored for reproducible future invocations. The filtered campaign uses a separate fixed launcher and source manifest.

## Completed filtered evolution

The filtered six-hour control closed at 21,600 s in 188.7 s of runtime. Hydrostatic, positivity, temperature and molecular-transport checks passed. Its peak saved upper vertical speed was 238.57 m/s at 412.5 km after 1200 s, versus 469.88 m/s for the identical unfiltered wind control. The maximum upper pressure change decreased from 136.73% to 49.86%; the maximum lower pressure change decreased from 15.42% to 6.98%. Six-hour final radial velocity at 387.5 km was 78.8 m/s. These are maxima over saved frames, not continuously sampled extrema.

Spatial filtering therefore reduces the startup transient by roughly half but does not eliminate it. It is not accepted as a balanced-weather production initialization. Digital-filter or another dynamical mass/wind initialization remains required; applying additional continuous wave damping simply to hide the startup transient would also alter subsequent physical waves.

Evidence: `data/era5_filter_t6_metal6h_validation.h5`, `data/era5_filter_t6_molecular_validation.h5`, `output/era5_isolation_v1/comparison_filtered/comparison.h5`. Comparison figure: `output/era5_isolation_v1/comparison_filtered/comparison.png`, generated by `scripts/compare_era5_isolation.py`. Source/input hashes in `output/era5_filter_t6_sources.sha256` matched after completion. No new 24-hour production run was launched.

## Gradual analysis insertion: Incremental Analysis Update

Bloom et al. (1996), *Data Assimilation Using Incremental Analysis Updates*,
MWR 124, 1256–1271, introduced IAU to distribute a fixed analysis increment
as a model tendency over an assimilation window:
https://doi.org/10.1175/1520-0493(1996)124%3C1256:DAUIAU%3E2.0.CO;2.
Our startup adaptation begins from the quiet hydrostatic reference and inserts
one spatially filtered ERA5 analysis over six hours. It is not repeated
assimilation or continuous relaxation toward a static weather map.

In `src/initialization.rs`, the fixed increment is the difference between the
hydrostatically integrated ERA5 target and the quiet state, in conserved
mass, ECEF momentum, and total gas-energy density. Horizontal winds are included;
vertical winds are not imposed. The increment is full strength below 30 km,
cosine tapered between 30 and 50 km, and exactly zero above 50 km. Tapering the
thermodynamic increment need not preserve exact hydrostatic balance; the slow
insertion permits adjustment by the dynamics rather than claiming an exactly
balanced final target.

For window length W, the cumulative fraction is
C(t)=x-sin(2*pi*x)/(2*pi), x=clamp(t/W,0,1).
The tendency in each SSPRK2 stage is
[C(t+dt)-C(t)]/dt times the fixed conserved increment. Thus the integrated
prescribed correction is exactly the increment, independent of step size.
All forcing is zero after W. This raised-cosine window is our adaptation of
Bloom's original constant-window scheme. HDF5 stores the full and applied
volume integrals of analysis mass, momentum, gas energy, and gravitational
potential energy separately from solar and rocket sources. These are externally
prescribed analysis corrections, not injected rocket material.

The unit tests cover irregular-step normalization, altitude exclusion, partition
additivity, and the window's response to a linear forced oscillator. They do not
establish atmospheric convergence. The 600-second CPU/Metal relative L2 maximum
is 8.493e-7; the two-rank CPU/serial difference is below 1e-20. See
`data/era5_iau_v1_cpu_metal_validation.h5` and
`data/era5_iau_v1_mpi_validation.h5`.

Becker et al. (2022), https://doi.org/10.1029/2021JD035018, use large-scale
spectral nudging in a whole-atmosphere model; that motivates restricting the
analysis to large horizontal scales and lower altitudes. This implementation
is finite IAU, not that model's live spectral relaxation. IAU is not a universal
stability guarantee: see *The Stability of Incremental Analysis Update* (2018),
https://journals.ametsoc.org/view/journals/mwre/146/10/mwr-d-18-0117.1.xml.

### Completed 12-hour control

`output/era5_iau_v1_metal12h` closed successfully with 73 frames, fixed 0.8333 s
steps, and 366.44 s compute time. The six-hour insertion was followed by six
hours of free evolution, with solar heating disabled. Same level-2 mesh,
52 levels to 450 km, rotation, molecular transport and absorbing boundaries
as the abrupt-initialization controls. All frozen source/input hashes match.

Peak vertical speed above 120 km: abrupt unfiltered ERA5 469.882 m/s over
six hours; spatially filtered abrupt ERA5 238.571 m/s over six hours; filtered
IAU 13.832 m/s during insertion and 26.280 m/s over the full twelve hours.
Maximum temperature was 1002.882 K. Maximum upper pressure departure from the
quiet reference was 14.472%, versus 136.726% for unfiltered abrupt insertion.
The lower pressure departure is not a wave-only measure: it includes the
intended ERA5 weather correction relative to the uniform reference.

The prescribed net analysis corrections integrated over the six-hour window
are +2.86890636e15 kg, ECEF momentum
(3.67665448e17,4.53490990e16,-3.15879438e16) kg m/s,
+9.43133224e20 J gas energy, and +1.32184704e21 J gravitational potential energy.
These are recorded external initialization corrections, not continuous heating
or a closed-atmosphere mass/energy conservation claim. Boundary fluxes, dynamics
and sponges also change inventories. No methane or oxygen was injected.

`validate_era5_iau.py` verifies closed frames, positivity, temperature, exact
normalized cumulative source metadata, and serial/MPI regression;
`validate_molecular_run.py` verifies transport coefficients and energy residuals.
Evidence: `data/era5_iau_v1_metal12h_validation.h5`,
`data/era5_iau_v1_molecular_validation.h5`.
`compare_era5_isolation.py` creates the comparison PNG/PDF/HDF5 in
`output/era5_iau_v1_comparison`. `export_viewer_layer.py` and
`render_era5_animation.py` generate the globe/meridional animations.

This suppresses the large startup shock. It does not prove long-term stability,
ERA5 forecast fidelity or thermospheric convergence; upper velocities increase
modestly during the free interval. Radiative cooling, turbulent closure and ion
drag remain absent. Repeat a longer quiet-control check before adding solar or
rocket forcing.

Run a new control (new output directory required):
`examples/era5/run_control.sh iau metal 43200 2`
Use `cpu` for the independent Rust CPU backend. Prepare the filtered ERA5 file
with `scripts/filter_era5_initial.py` first. `--era5-iau-seconds 21600` enables
IAU; zero retains the existing direct-initialization path.
