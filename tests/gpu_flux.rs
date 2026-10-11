use global_neutral_euler::{compute::*, mesh::*, parallel::Domain, physics::*, solver::*};

fn gpu() -> Option<Accelerator> {
    let accelerator=Accelerator::discover("gpu",Precision::F32,0).unwrap();
    if std::env::var_os("RNS_REQUIRE_GPU").is_some() {assert!(accelerator.is_some(),"GPU required for this test run");}
    accelerator
}
#[test]
fn gpu_flux_matches_f64_across_backgrounds_and_amplitudes() {
    let Some(accelerator)=gpu() else {eprintln!("No GPU: comparison skipped");return;};
    let mut faces=vec![];
    for rho in [1.,1e-8,1e-15] {
        let a=Atmos{rho,p:rho*86100.,r:287.,cv:717.5,g:9.81};
        let b=Atmos{rho:rho*0.8,p:rho*86100.*0.9,..a};
        for amplitude in [0.,1e-9,1e-6,0.01,0.5] {
            for mixture in [false,true] {
                let ch=if mixture {0.02}else{0.};
                let l=conserved([1.+amplitude,300.*amplitude,40.*amplitude,0.,1.+amplitude,ch,3.6*ch],a);
                let r=conserved([1.-amplitude/2.,-20.*amplitude,0.,40.*amplitude,1.-amplitude/3.,ch/2.,1.8*ch],b);
                faces.push(Face{left:l,right:r,al:a,ar:b,normal:unit([1.,2.,3.]),fraction:0.3});
                faces.push(Face{left:base(a),right:base(b),al:a,ar:b,normal:unit([1.,2.,3.]),fraction:0.3});
            }
        }
    }
    let actual=accelerator.fluxes(&faces).unwrap();
    for (face,got) in faces.iter().zip(actual) {
        let expected=face.cpu();
        for f in 0..7 {
            let scale=if f==4 {face.al.p*1000.}else if f>0 && f<4 {face.al.p}else{face.al.rho*1000.};
            assert!((got[f]-expected[f]).abs()<4e-5*expected[f].abs()+2e-14*scale,"field {f}: {} != {}",got[f],expected[f]);
        }
    }
}
#[test]
fn gpu_rhs_matches_cpu_and_preserves_rest() {
    let Some(accelerator)=gpu() else {return;};
    let mut m=Model::isothermal(2,12,5000.);m.sponge=true;m.rotation=true;
    let d=Domain::new(&m.mesh,0,1);
    let mut work=RhsWorkspace::new(&m,&d);work.accelerator=Some(accelerator);
    let rest=d.needed.iter().flat_map(|_|m.background.iter().map(|a|base(*a))).collect::<Vec<_>>();
    assert!(m.rhs_cached(&d,&rest,&mut work).iter().flatten().all(|v|*v==0.));
    for amplitude in [1e-9,1e-4,0.1] {
        let mut u=rest.clone();
        for &i in &d.needed {for k in 0..12 {
            let x=amplitude*m.mesh.cells[i].center[0]*(k as f64*0.3).sin();
            u[d.index(i,k,12)]=conserved([1.+x,100.*x,20.*x,0.,1.+x,0.,0.],m.background[k]);
        }}
        let mut cpu=RhsWorkspace::new(&m,&d);let expected=m.rhs_cached(&d,&u,&mut cpu).to_vec();let actual=m.rhs_cached(&d,&u,&mut work);
        for f in 0..7 {
            let scale=expected.iter().map(|q|q[f].abs()).fold(0f64,f64::max);
            let error=expected.iter().zip(actual.iter()).map(|(a,b)|(a[f]-b[f]).abs()).fold(0f64,f64::max);
            let tolerance=if amplitude<1e-8 {2e-3}else{2e-4};
            assert!(error<=tolerance*scale.max(1e-30),"amplitude {amplitude}, field {f}: relative error {}",error/scale);
        }
    }
}
#[test]
fn explicit_cpu_never_initializes_gpu() {
    assert!(Accelerator::discover("cpu",Precision::F64,0).unwrap().is_none());
    assert!(Accelerator::discover("invalid",Precision::F32,0).is_err());
}

#[test]
fn stretched_space_rhs_matches_cpu() {
 let Some(accelerator)=gpu() else {return;};
 let f=hdf5::File::open("data/era5_msis_20250306_23utc.h5").unwrap();
 let read=|n:&str|f.dataset(n).unwrap().read_raw::<f64>().unwrap();
 let faces=read("altitude_faces_m");let nz=faces.len()-1;
 let mut m=Model::isothermal(1,nz,2000.);m.set_vertical(global_neutral_euler::vertical::VerticalGrid::new(faces));m.sponge=true;m.rotation=true;
 let rho=read("density_kg_m3");let p=read("pressure_pa");let r=read("gas_constant_j_kg_k");let cv=read("cv_j_kg_k");let g=read("gravity_m_s2");
 for k in 0..nz {m.background[k]=Atmos{rho:rho[k],p:p[k],r:r[k],cv:cv[k],g:g[k]};}
 m.discrete_hydrostatic=true;let d=Domain::new(&m.mesh,0,1);let e=global_neutral_euler::era5::Era5::load("data/era5_20250306_23utc.h5").unwrap();let u=e.initialize(&m,&d);
 assert!(d.needed.iter().all(|&i|(0..nz).all(|k|admissible(u[d.index(i,k,nz)],m.background[k]))));
 let mut cpu=RhsWorkspace::new(&m,&d);let expected=m.rhs_cached(&d,&u,&mut cpu).to_vec();let mut work=RhsWorkspace::new(&m,&d);work.accelerator=Some(accelerator);let actual=m.rhs_cached(&d,&u,&mut work);
 // Compare each altitude separately so dense lower layers cannot hide upper errors.
 for k in 0..nz {for field in 0..5 {let mut scale=0f64;let mut error=0f64;for &i in &d.owned {let j=d.index(i,k,nz);scale=scale.max(expected[j][field].abs());error=error.max((expected[j][field]-actual[j][field]).abs());}assert!(error<2e-3*scale.max(1e-30),"layer {k}, field {field}: {}",error/scale);}}
}

#[test]
fn scaled_hydrostatic_column_is_balanced_on_stretched_grid() {
 let mut m=Model::isothermal(2,149,2500.);m.set_vertical(global_neutral_euler::vertical::VerticalGrid::starship_stretched());m.discrete_hydrostatic=true;let nz=m.background.len();let d=Domain::new(&m.mesh,0,1);
 let u=d.needed.iter().flat_map(|_|m.background.iter().map(|a|conserved([1.1,0.,0.,0.,1.1,0.,0.],*a))).collect::<Vec<_>>();let rhs=m.rhs(&d,&u);
 for &i in &d.owned {for k in 1..nz-1 {let q=rhs[d.index(i,k,nz)];let scale=m.background[k].rho*m.background[k].g;assert!(norm([q[1],q[2],q[3]])<1e-10*scale,"layer {k}: {:?}",q);}}
}
#[test]
fn stretched_temperature_reconstruction_matches_cpu() {
 let Some(accelerator)=gpu() else {return;};
 let f=hdf5::File::open("data/era5_msis_20250306_23utc.h5").unwrap();
 let read=|n:&str|f.dataset(n).unwrap().read_raw::<f64>().unwrap();
 let faces=read("altitude_faces_m");let nz=faces.len()-1;
 let mut m=Model::isothermal(1,nz,2000.);m.set_vertical(global_neutral_euler::vertical::VerticalGrid::new(faces));m.sponge=true;m.rotation=true;
 let rho=read("density_kg_m3");let p=read("pressure_pa");let r=read("gas_constant_j_kg_k");let cv=read("cv_j_kg_k");let g=read("gravity_m_s2");
 for k in 0..nz {m.background[k]=Atmos{rho:rho[k],p:p[k],r:r[k],cv:cv[k],g:g[k]};}
 m.reconstruct_temperature=true;m.discrete_hydrostatic=true;let d=Domain::new(&m.mesh,0,1);let e=global_neutral_euler::era5::Era5::load("data/era5_20250306_23utc.h5").unwrap();let u=e.initialize(&m,&d);
 assert!(d.needed.iter().all(|&i|(0..nz).all(|k|admissible(u[d.index(i,k,nz)],m.background[k]))));
 let mut cpu=RhsWorkspace::new(&m,&d);let expected=m.rhs_cached(&d,&u,&mut cpu).to_vec();let mut work=RhsWorkspace::new(&m,&d);work.accelerator=Some(accelerator);let actual=m.rhs_cached(&d,&u,&mut work);
 // Compare each altitude separately so dense lower layers cannot hide upper errors.
 for k in 0..nz {for field in 0..5 {let mut scale=0f64;let mut error=0f64;for &i in &d.owned {let j=d.index(i,k,nz);scale=scale.max(expected[j][field].abs());error=error.max((expected[j][field]-actual[j][field]).abs());}assert!(error<2e-3*scale.max(1e-30),"layer {k}, field {field}: {}",error/scale);}}
}

