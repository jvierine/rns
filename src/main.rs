use anyhow::{Result, ensure};
use global_neutral_euler::{mesh::*, parallel::Domain, physics::*, solver::{Model,RhsWorkspace}};
use mpi::{collective::SystemOperation, traits::*};
use std::{path::PathBuf, time::Instant};
fn scalar(g: &hdf5::Group, name: &str, value: f64) -> Result<()> {
    g.new_attr::<f64>().create(name)?.write_scalar(&value)?;
    Ok(())
}
fn data(g: &hdf5::Group, name: &str, values: &[f64]) -> Result<()> {
    g.new_dataset_builder().with_data(values).deflate(1).create(name)?;
    Ok(())
}
fn main() -> Result<()> {
    let universe = mpi::initialize().expect("MPI initialization");
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let get = |key: &str, default: &str| {
        args.windows(2)
            .find(|w| w[0] == key)
            .map(|w| w[1].clone())
            .unwrap_or(default.into())
    };
    let level = get("--level", "3").parse::<usize>()?;
    let nz = get("--layers", "12").parse::<usize>()?;
    let dz = get("--dz", "5000").parse::<f64>()?;
    let end = get("--end", "600").parse::<f64>()?;
    let cadence = get("--cadence", "30").parse::<f64>()?;
    let sigma = get("--sigma-km", "1000").parse::<f64>()? * 1000.;
    let case = get("--case", "lamb");
    let background_path = get("--background", "");
    let directory = PathBuf::from(get("--output", "global_icosahedral_lamb"));
    ensure!(
        nz >= 3
            && dz > 0.
            && end >= 0.
            && cadence > 0.
            && sigma.is_finite() && sigma > 0.
            && ["lamb", "heat", "equilibrium"].contains(&case.as_str()),
        "Invalid configuration"
    );
    if rank == 0 {
        if directory.exists() {
            eprintln!("Refuse existing output directory");
            world.abort(4);
        }
        std::fs::create_dir_all(&directory)?;
    }
    world.barrier();
    let mut model = Model::isothermal(level, nz, dz);
    if !background_path.is_empty() {
        let bg = hdf5::File::open(&background_path)?;
        let fields = ["density_kg_m3", "pressure_pa", "gas_constant_j_kg_k", "cv_j_kg_k", "gravity_m_s2", "altitude_m"]
            .map(|name| bg.dataset(name).and_then(|d| d.read_raw::<f64>()));
        let mut v = Vec::new();
        for field in fields { v.push(field?); }
        ensure!(v.iter().all(|a| a.len()==nz), "Background dimension mismatch");
        for k in 0..nz {
            ensure!((v[5][k]-(k as f64+0.5)*dz).abs()<1e-6 && v[..5].iter().all(|a| a[k].is_finite() && a[k]>0.), "Invalid background");
            model.background[k] = Atmos { rho:v[0][k], p:v[1][k], r:v[2][k], cv:v[3][k], g:v[4][k] };
        }
        if rank==0 { std::fs::copy(&background_path, directory.join("background_input.h5"))?; }
    }
    model.initial_sigma_m = sigma;
    model.heat_pulse = case == "heat";
    model.sponge = args.contains(&"--sponge".to_string());
    model.rotation = args.contains(&"--rotation".to_string());
    model.second_order = !args.contains(&"--first-order".to_string());
    let domain = Domain::new(&model.mesh, rank, size);
    let mut u = model.initial(&domain, case != "equilibrium");
    let mut stage = u.clone();
    let mut work = RhsWorkspace::new(&model,&domain);
    let mut exchange = domain.exchange_workspace(nz);
    let file = hdf5::File::create(directory.join(format!("rank_{rank:04}.h5")))?;
    let root = file.group("/")?;
    scalar(&root, "run_complete", 0.)?;
    scalar(&root, "rank", rank as f64)?;
    scalar(&root, "mpi_size", size as f64)?;
    scalar(&root, "dz_m", dz)?;
    scalar(&root, "layers", nz as f64)?;
    scalar(&root, "end_time_s", end)?;
    scalar(&root, "second_order", model.second_order as u8 as f64)?;
    scalar(&root, "sponge", model.sponge as u8 as f64)?;
    scalar(&root, "rotation", model.rotation as u8 as f64)?;
    scalar(&root, "initial_lamb_mode", (case == "lamb") as u8 as f64)?;
    scalar(&root, "initial_heat_pulse", model.heat_pulse as u8 as f64)?;
    data(&root, "background_gas_constant_j_kg_k", &model.background.iter().map(|a| a.r).collect::<Vec<_>>())?;
    data(&root, "background_cv_j_kg_k", &model.background.iter().map(|a| a.cv).collect::<Vec<_>>())?;
    data(
        &root,
        "background_density_kg_m3",
        &model.background.iter().map(|a| a.rho).collect::<Vec<_>>(),
    )?;
    data(
        &root,
        "background_pressure_pa",
        &model.background.iter().map(|a| a.p).collect::<Vec<_>>(),
    )?;
    data(
        &root,
        "global_column_id",
        &domain.owned.iter().map(|&i| i as f64).collect::<Vec<_>>(),
    )?;
    let frames = file.create_group("frames")?;
    if rank == 0 {
        let mesh = hdf5::File::create(directory.join("mesh.h5"))?;
        let g = mesh.group("/")?;
        data(
            &g,
            "vertices_unit_xyz",
            &model
                .mesh
                .vertices
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )?;
        data(
            &g,
            "triangle_vertex_indices",
            &model
                .mesh
                .cells
                .iter()
                .flat_map(|c| c.vertices.map(|v| v as f64))
                .collect::<Vec<_>>(),
        )?;
        data(
            &g,
            "center_unit_xyz",
            &model
                .mesh
                .cells
                .iter()
                .flat_map(|c| c.center)
                .collect::<Vec<_>>(),
        )?;
        data(
            &g,
            "solid_angle_sr",
            &model.mesh.cells.iter().map(|c| c.area).collect::<Vec<_>>(),
        )?;
        scalar(&g, "earth_radius_m", RE)?;
        scalar(&g, "refinement_level", level as f64)?;
        scalar(&g, "sound_speed_m_s", (1.4 * 287f64 * 300.).sqrt())?;
        scalar(&g, "source_latitude_deg", -20.54)?;
        scalar(&g, "source_longitude_deg", -175.38)?;
        scalar(&g, "initial_sigma_m", sigma)?;
        scalar(&g, "initial_surface_pressure_pa", 1e-7 * 287. * 300.)?;
        scalar(&g, "initial_heat_pulse", model.heat_pulse as u8 as f64)?;
        scalar(&g, "variable_background", (!background_path.is_empty()) as u8 as f64)?;
        scalar(&g, "initial_vertical_center_m", 10_000.)?;
        scalar(&g, "initial_vertical_sigma_m", 5_000.)?;
        println!(
            "columns={} layers={} cells={} ranks={size} case={case}",
            model.mesh.cells.len(),
            nz,
            model.mesh.cells.len() * nz
        );
    }
    let start = Instant::now();
    let mut t = 0.;
    let mut next = 0.;
    let mut frame = 0;
    let mut steps = 0;
    loop {
        if t >= next - 1e-8 || t >= end - 1e-8 {
            let group = frames.create_group(&format!("{frame:05}"))?;
            scalar(&group, "time_s", t)?;
            let mut state = Vec::new();
            for &i in &domain.owned {
                for k in 0..nz {
                    state.extend(u[domain.index(i, k, nz)]);
                }
            }
            data(
                &group,
                "conserved_density_ecef_momentum_energy_ch4_o2",
                &state,
            )?;
            let pp = domain
                .owned
                .iter()
                .map(|&i| {
                    thermo(u[domain.index(i, 0, nz)], model.background[0]).0 - model.background[0].p
                })
                .collect::<Vec<_>>();
            data(&group, "surface_pressure_perturbation_pa", &pp)?;
            let mut horizontal = Vec::new();
            let mut radial = Vec::new();
            for &i in &domain.owned {
                let mut sum = [0.;3]; let mut count = 0.;
                for k in 0..nz {
                    let z=(k as f64+0.5)*dz;
                    if z>=80_000. && z<=100_000. {
                        let q=u[domain.index(i,k,nz)];
                        for c in 0..3 { sum[c]+=q[c+1]/q[0]; }
                        count+=1.;
                    }
                }
                if count>0. { for x in &mut sum { *x/=count; } }
                let up=model.mesh.cells[i].center; let vr=dot(sum,up);
                radial.push(vr);
                horizontal.extend(sub(sum,mul(up,vr)));
            }
            data(&group,"mesosphere_80_100km_horizontal_velocity_ecef_m_s", &horizontal)?;
            data(&group,"mesosphere_80_100km_radial_velocity_m_s", &radial)?;
            let mut inv = [0.; 7];
            for &i in &domain.owned {
                for k in 0..nz {
                    let vol = model.mesh.volume(i, k, dz);
                    for f in 0..7 {
                        inv[f] += u[domain.index(i, k, nz)][f] * vol;
                    }
                }
            }
            let mut total = [0.; 7];
            world.all_reduce_into(&inv, &mut total, SystemOperation::sum());
            data(&group, "global_volume_integrals", &total)?;
            // Commit rank metadata/data at every saved frame. A paused, copied
            // archive can then be checked without waiting for final closure.
            // This source change does not modify an already-running binary.
            file.flush()?;
            if rank == 0 {
                println!(
                    "t={t:0.3} steps={steps} wall={:0.3}",
                    start.elapsed().as_secs_f64()
                );
            }
            frame += 1;
            next += cadence;
            if t >= end - 1e-8 {
                break;
            }
        }
        let local = model.local_dt(&domain, &u);
        let mut dt = 0f64;
        world.all_reduce_into(&local, &mut dt, SystemOperation::min());
        dt = dt.min(next - t).min(end - t);
        ensure!(dt > 0. && dt.is_finite(), "Invalid time step");
        domain.exchange_cached(&world, &mut u, nz, &mut exchange);
        let rhs = model.rhs_cached(&domain, &u, &mut work);
        stage.copy_from_slice(&u);
        for &i in &domain.owned {
            for k in 0..nz {
                let n = domain.index(i, k, nz);
                for f in 0..7 {
                    stage[n][f] += dt * rhs[n][f];
                }
            }
        }
        let local_ok = domain.owned.iter().all(|&i| {
            (0..nz).all(|k| admissible(stage[domain.index(i, k, nz)], model.background[k]))
        }) as i32;
        let mut all_ok = 0;
        world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
        if all_ok == 0 {
            if rank == 0 {
                eprintln!("Nonpositive first RK stage; refusing to continue");
            }
            world.abort(2);
        }
        domain.exchange_cached(&world, &mut stage, nz, &mut exchange);
        let rhs = model.rhs_cached(&domain, &stage, &mut work);
        for &i in &domain.owned {
            for k in 0..nz {
                let n = domain.index(i, k, nz);
                for f in 0..7 {
                    u[n][f] = 0.5 * (u[n][f] + stage[n][f] + dt * rhs[n][f]);
                }
            }
        }
        let local_ok =
            domain.owned.iter().all(|&i| {
                (0..nz).all(|k| admissible(u[domain.index(i, k, nz)], model.background[k]))
            }) as i32;
        world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
        if all_ok == 0 {
            world.abort(3);
        }
        t += dt;
        steps += 1;
    }
    file.attr("run_complete")?.write_scalar(&1f64)?;
    file.flush()?;
    drop(frames);
    drop(root);
    drop(file);
    world.barrier();
    if rank == 0 {
        std::fs::write(
            directory.join("run.closed"),
            b"All rank archives closed after successful simulation\n",
        )?;
    }
    Ok(())
}
