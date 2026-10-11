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
fn check_temperature(m:&Model,d:&Domain,w:&impl CommunicatorCollectives,u:&[Q],limit:f64,t:f64,phase:&str)->Result<()> {
    if limit==0. {return Ok(());}
    let nz=m.background.len();let mut local=0f64;
    for &i in &d.owned {for k in 0..nz {let value=temperature(u[d.index(i,k,nz)],m.background[k]);if !value.is_finite()||value<=0. {local=f64::INFINITY;}else{local=local.max(value);}}}
    let mut peak=0f64;w.all_reduce_into(&local,&mut peak,SystemOperation::max());
    ensure!(peak<=limit,"Temperature safety gate: {peak} K exceeds {limit} K at t={t} ({phase}); state was not clipped");Ok(())
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
    let temperature_limit=get("--max-temperature-k","0").parse::<f64>()?;ensure!(temperature_limit.is_finite() && temperature_limit>=0.,"Invalid temperature safety limit");
    let cfl=get("--cfl","0.3").parse::<f64>()?;ensure!(cfl>0. && cfl<=0.3,"CFL must be in (0,0.3]");
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
        if bg.link_exists("altitude_faces_m") {
            let faces=bg.dataset("altitude_faces_m")?.read_raw::<f64>()?;
            ensure!(faces.len()==nz+1 && faces[0]==0. && faces.iter().all(|x|x.is_finite()) && faces.windows(2).all(|w|w[1]>w[0]), "Invalid altitude faces");
            model.set_vertical(global_neutral_euler::vertical::VerticalGrid::new(faces));
        }
        let fields = ["density_kg_m3", "pressure_pa", "gas_constant_j_kg_k", "cv_j_kg_k", "gravity_m_s2", "altitude_m"]
            .map(|name| bg.dataset(name).and_then(|d| d.read_raw::<f64>()));
        let mut v = Vec::new();
        for field in fields { v.push(field?); }
        ensure!(v.iter().all(|a| a.len()==nz), "Background dimension mismatch");
        for k in 0..nz {
            ensure!((v[5][k]-model.vertical.centers[k]).abs()<1e-6 && v[..5].iter().all(|a| a[k].is_finite() && a[k]>0.), "Invalid background");
            model.background[k] = Atmos { rho:v[0][k], p:v[1][k], r:v[2][k], cv:v[3][k], g:v[4][k] };
        }
        if rank==0 { std::fs::copy(&background_path, directory.join("background_input.h5"))?; }
    }
    model.initial_sigma_m = sigma;
    model.heat_pulse = case == "heat";
    model.sponge = args.contains(&"--sponge".to_string());
    model.conservative_gravity_energy=args.contains(&"--conservative-gravity-energy".to_string());
    model.discrete_hydrostatic=args.contains(&"--discrete-hydrostatic".to_string());
    model.rotation = args.contains(&"--rotation".to_string());
    model.reconstruct_temperature=args.contains(&"--reconstruct-temperature".to_string());
    model.second_order = !args.contains(&"--first-order".to_string());
    let domain = Domain::new(&model.mesh, rank, size);
    let era5_path=get("--era5","");
    let era5=if era5_path.is_empty(){None}else{Some(global_neutral_euler::era5::Era5::load(&era5_path)?)};
    let iau_duration=get("--era5-iau-seconds","0").parse::<f64>()?;
    ensure!(iau_duration.is_finite()&&iau_duration>=0.,"Invalid IAU duration");
    let column_hydrostatic=args.contains(&"--era5-hydrostatic-columns".to_string());
    ensure!(!column_hydrostatic || (era5.is_some() && model.discrete_hydrostatic),"Hydrostatic columns require ERA5 and discrete hydrostatic gravity");
    if column_hydrostatic{model.column_background=Some(era5.as_ref().unwrap().hydrostatic_columns(&model));}
    let uniform_upper=args.contains(&"--era5-uniform-upper".to_string());
    let hydrostatic_upper=args.contains(&"--era5-hydrostatic-join".to_string());
    let upper_start=get("--era5-upper-start-km","50").parse::<f64>()?*1000.;
    let upper_end=get("--era5-upper-end-km","120").parse::<f64>()?*1000.;
    ensure!(!uniform_upper || (upper_start.is_finite() && upper_end.is_finite() && upper_start>=30000. && upper_end>upper_start),"Invalid upper initialization blend");
    ensure!(!hydrostatic_upper || (uniform_upper && upper_start>=50000. && era5.is_some()),"Hydrostatic join requires ERA5, uniform upper, and start >=50 km");
    ensure!(!(column_hydrostatic && (uniform_upper || hydrostatic_upper)),"Choose column hydrostatic initialization or uniform-upper control");
    ensure!(iau_duration==0. || (era5.is_some() && !column_hydrostatic && !uniform_upper && !hydrostatic_upper && case=="equilibrium"),"IAU requires ERA5 and equilibrium; do not combine with direct ERA5 initialization");
    let iau=if iau_duration>0. {Some(global_neutral_euler::initialization::AnalysisUpdate::new(&model,&domain,era5.as_ref().unwrap(),iau_duration,get("--era5-iau-lower-km","30").parse::<f64>()?*1000.,get("--era5-iau-upper-km","50").parse::<f64>()?*1000.)?)}else{None};
    let mut u=if iau.is_some(){model.initial(&domain,false)}else if let Some(e)=&era5 {if column_hydrostatic{e.initialize_hydrostatic_columns(&model,&domain,!args.contains(&"--era5-no-winds".to_string()))}else if hydrostatic_upper{e.initialize_hydrostatic_upper(&model,&domain,upper_start,upper_end)}else if uniform_upper{e.initialize_uniform_upper(&model,&domain,upper_start,upper_end)}else{e.initialize(&model,&domain)}}else{model.initial(&domain,case!="equilibrium")};
    let solar=get("--solar-absorption","0").parse::<f64>()?;ensure!(solar>=0. && solar<=1.,"Solar absorption");
    let epoch=era5.as_ref().map(|e|e.epoch).unwrap_or(0.);
    let mut stage = u.clone();
    let mut work = RhsWorkspace::new(&model,&domain);
    let precision=global_neutral_euler::compute::Precision::parse(&get("--precision","f32"))?;
    let backend=get("--backend","auto");
    let device_index=get("--device","0").parse::<usize>()?;
    work.accelerator=global_neutral_euler::compute::Accelerator::discover(&backend,precision,device_index)?;
    let local_gpu=work.accelerator.is_some() as i32;
    let mut all_gpu=0;
    world.all_reduce_into(&local_gpu,&mut all_gpu,SystemOperation::min());
    if all_gpu==0 {work.accelerator=None;}
    eprintln!("rank {rank}: {}",work.accelerator.as_ref().map(|a|a.label.as_str()).unwrap_or("CPU / f64"));
    let transport_path=get("--transport","");
    let transport_interval=get("--transport-interval","5").parse::<f64>()?;
    let mut transport=if transport_path.is_empty(){None}else{
        let f=hdf5::File::open(&transport_path)?;
        ensure!(f.dataset("altitude_m")?.read_raw::<f64>()?==model.vertical.centers,"Transport coordinates mismatch");
        let read=|n:&str|->Result<Vec<f64>>{Ok(f.dataset(&format!("transport/{n}"))?.read_raw::<f64>()?)};
        let t0=read("background_temperature_k")?;
        ensure!(t0.len()==nz,"Transport temperature shape mismatch");
        ensure!(t0.iter().enumerate().all(|(k,t)|(*t-model.background[k].p/(model.background[k].rho*model.background[k].r)).abs()<1e-8*t),"Transport/background temperature mismatch");
        Some(global_neutral_euler::transport::Transport::new(&model,&domain,read("dynamic_viscosity_pa_s")?,read("thermal_conductivity_w_m_k")?,t0,transport_interval)?)
    };
    let mut iau_integrals=[0.;8];
    if let Some(a)=&iau {world.all_reduce_into(&a.integrals(&model,&domain),&mut iau_integrals,SystemOperation::sum());}
    let mut exchange = domain.exchange_workspace(nz);
    let file = hdf5::File::create(directory.join(format!("rank_{rank:04}.h5")))?;
    let root = file.group("/")?;
    scalar(&root,"gpu_flux_enabled",work.accelerator.is_some() as u8 as f64)?;
    scalar(&root,"flux_precision_bits",if work.accelerator.is_none() || precision==global_neutral_euler::compute::Precision::F64 {64.}else{32.})?;
    scalar(&root,"state_precision_bits",64.)?;
    scalar(&root,"temperature_reconstruction",model.reconstruct_temperature as u8 as f64)?;
    scalar(&root,"maximum_temperature_gate_k",temperature_limit)?;
    scalar(&root, "run_complete", 0.)?;
    scalar(&root,"molecular_transport_enabled",transport.is_some()as u8 as f64)?;
    if let Some(tr)=&transport {
        let group=file.create_group("transport")?;
        data(&group,"dynamic_viscosity_pa_s",&tr.mu)?;data(&group,"thermal_conductivity_w_m_k",&tr.kappa)?;data(&group,"background_temperature_k",&tr.t0)?;scalar(&group,"interval_s",tr.interval)?;
        if rank==0 {std::fs::copy(&transport_path,directory.join("transport_input.h5"))?;}
    }
    scalar(&root,"era5_iau_window_s",iau_duration)?;
    if let Some(a)=&iau {scalar(&root,"era5_iau_lower_m",a.lower)?;scalar(&root,"era5_iau_upper_m",a.upper)?;data(&root,"iau_full_increment_volume_integrals",&iau_integrals)?;}
    if rank==0 && era5.is_some(){std::fs::copy(&era5_path,directory.join("era5_input.h5"))?;}
    scalar(&root,"era5_uniform_upper",uniform_upper as u8 as f64)?;
    scalar(&root,"era5_hydrostatic_join",hydrostatic_upper as u8 as f64)?;
    scalar(&root,"era5_hydrostatic_columns",column_hydrostatic as u8 as f64)?;
    scalar(&root,"hydrostatic_column_flux_scheme",2.)?;
    scalar(&root,"hydrostatic_column_initialization",3.)?;
    if column_hydrostatic{let density:Vec<_>=domain.owned.iter().flat_map(|&i|(0..nz).map(|k|model.at(i,k).rho).collect::<Vec<_>>()).collect();let pressure:Vec<_>=domain.owned.iter().flat_map(|&i|(0..nz).map(|k|model.at(i,k).p).collect::<Vec<_>>()).collect();data(&root,"column_reference_density_kg_m3",&density)?;data(&root,"column_reference_pressure_pa",&pressure)?;}
    scalar(&root,"era5_upper_blend_start_m",upper_start)?;scalar(&root,"era5_upper_blend_end_m",upper_end)?;
    scalar(&root,"transport_affine_face_stabilization",1.)?;
    scalar(&root,"reference_unix_s",epoch)?;scalar(&root,"solar_absorbed_fraction",solar)?;
    scalar(&root, "rank", rank as f64)?;
    scalar(&root, "mpi_size", size as f64)?;
    scalar(&root, "dz_m", model.vertical.uniform_spacing().unwrap_or(0.))?;
    data(&root,"altitude_m",&model.vertical.centers)?;
    data(&root,"altitude_faces_m",&model.vertical.faces)?;
    scalar(&root, "layers", nz as f64)?;
    scalar(&root, "end_time_s", end)?;
    scalar(&root, "second_order", model.second_order as u8 as f64)?;
    scalar(&root, "sponge", model.sponge as u8 as f64)?;
    scalar(&root, "rotation", model.rotation as u8 as f64)?;
    scalar(&root,"conservative_gravity_energy",model.conservative_gravity_energy as u8 as f64)?;
    scalar(&root,"discrete_hydrostatic",model.discrete_hydrostatic as u8 as f64)?;
    scalar(&root,"cfl",cfl)?;
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
    let initial_gpu=all_gpu;
    let start = Instant::now();
    let mut t = 0.;
    let mut next = 0.;
    let mut frame = 0;
    let mut steps = 0;
    let mut last_transport=0.;let mut next_transport=transport_interval;
    loop {
        if transport.is_some() && t>last_transport && (t>=next_transport-1e-8 || t>=end-1e-8) {
            transport.as_mut().unwrap().advance(&model,&domain,&world,&mut u,t-last_transport)?;
            last_transport=t;next_transport+=transport_interval;
        }
        check_temperature(&model,&domain,&world,&u,temperature_limit,t,"initial / after transport")?;
        if t >= next - 1e-8 || t >= end - 1e-8 {
            let group = frames.create_group(&format!("{frame:05}"))?;
            scalar(&group, "time_s", t)?;
            if let Some(a)=&iau{scalar(&group,"iau_applied_fraction",a.fraction(t))?;data(&group,"iau_applied_volume_integrals",&iau_integrals.map(|x|x*a.fraction(t)))?;}
            if let Some(tr)=&transport {scalar(&group,"viscous_kinetic_loss_j",tr.viscous_heat_j)?;scalar(&group,"viscous_total_energy_residual_j",tr.viscous_work_residual_j)?;scalar(&group,"conduction_total_energy_residual_j",tr.conduction_residual_j)?;scalar(&group,"transport_max_residual",tr.max_residual)?;scalar(&group,"transport_max_iterations",tr.max_iterations as f64)?;}
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
                    thermo(u[domain.index(i, 0, nz)], model.at(i,0)).0 - model.at(i,0).p
                })
                .collect::<Vec<_>>();
            data(&group, "surface_pressure_perturbation_pa", &pp)?;
            let mut horizontal = Vec::new();
            let mut radial = Vec::new();
            for &i in &domain.owned {
                let mut sum = [0.;3]; let mut count = 0.;
                for k in 0..nz {
                    let z=model.vertical.centers[k];
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
                    let vol = model.volume(i, k);
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
        let local = model.local_dt(&domain, &u)*(cfl/0.3);
        let mut dt = 0f64;
        world.all_reduce_into(&local, &mut dt, SystemOperation::min());
        dt = dt.min(next - t).min(end - t);
        if transport.is_some(){dt=dt.min(next_transport-t);}
        ensure!(dt > 0. && dt.is_finite(), "Invalid time step");
        domain.exchange_cached(&world, &mut u, nz, &mut exchange);
        model.rhs_cached(&domain, &u, &mut work);
        if all_gpu!=0 {
            let available=work.accelerator.is_some()as i32;
            world.all_reduce_into(&available,&mut all_gpu,SystemOperation::min());
            if all_gpu==0 {work.accelerator=None;model.rhs_cached(&domain,&u,&mut work);}
        }
        let solar_rhs=if solar>0. {Some(global_neutral_euler::era5::solar_rates(&model,&domain,epoch+t+dt/2.,solar))}else{None};
        let iau_rate=iau.as_ref().map(|a|a.rate(t,dt)).unwrap_or(0.);
        let rhs=work.result();
        stage.copy_from_slice(&u);
        for &i in &domain.owned {
            for k in 0..nz {
                let n = domain.index(i, k, nz);
                for f in 0..7 {
                    stage[n][f] += dt * (rhs[n][f]+solar_rhs.as_ref().map(|v|v[n][f]).unwrap_or(0.)+iau.as_ref().map(|a|iau_rate*a.increment[n][f]).unwrap_or(0.));
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
                eprintln!("Nonpositive first RK stage; refusing to continue at t={t}, dt={dt}");
                for &i in &domain.owned { for k in 0..nz {let n=domain.index(i,k,nz); if !admissible(stage[n],model.background[k]) {eprintln!("column={i} layer={k} altitude={} q={:?} pressure={}",model.vertical.centers[k],stage[n],thermo(stage[n],model.background[k]).0);break;} }}
            }
            world.abort(2);
        }
        check_temperature(&model,&domain,&world,&stage,temperature_limit,t+dt,"first RK stage")?;
        domain.exchange_cached(&world, &mut stage, nz, &mut exchange);
        model.rhs_cached(&domain, &stage, &mut work);
        if all_gpu!=0 {
            let available=work.accelerator.is_some()as i32;
            world.all_reduce_into(&available,&mut all_gpu,SystemOperation::min());
            if all_gpu==0 {work.accelerator=None;model.rhs_cached(&domain,&stage,&mut work);}
        }
        let solar_rhs=if solar>0. {Some(global_neutral_euler::era5::solar_rates(&model,&domain,epoch+t+dt/2.,solar))}else{None};
        let iau_rate=iau.as_ref().map(|a|a.rate(t,dt)).unwrap_or(0.);
        let rhs=work.result();
        for &i in &domain.owned {
            for k in 0..nz {
                let n = domain.index(i, k, nz);
                for f in 0..7 {
                    u[n][f] = 0.5 * (u[n][f] + stage[n][f] + dt * (rhs[n][f]+solar_rhs.as_ref().map(|v|v[n][f]).unwrap_or(0.)+iau.as_ref().map(|a|iau_rate*a.increment[n][f]).unwrap_or(0.)));
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
        check_temperature(&model,&domain,&world,&u,temperature_limit,t+dt,"second RK stage")?;
        t += dt;
        steps += 1;
    }
    scalar(&root,"gpu_runtime_fallback",(initial_gpu!=0 && all_gpu==0)as u8 as f64)?;
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
