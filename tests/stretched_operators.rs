use global_neutral_euler::{
    mesh::*,
    parallel::Domain,
    physics::*,
    solver::{Model, RhsWorkspace},
    vertical::VerticalGrid,
};
#[test]
fn cached_flux_reference_all_fields() {
    for stretched in [false, true] {
        for strong in [false, true] {
            let mut m = Model::isothermal(2, 12, 2500.);
            if stretched {
                m.set_vertical(VerticalGrid::new(vec![
                    0., 1000., 2000., 4000., 8000., 12000., 20000., 30000.,
                ]));
            }
            let d = Domain::new(&m.mesh, 0, 1);
            let nz = m.background.len();
            let mut u = m.initial(&d, false);
            for &i in &d.owned {
                for k in 0..nz {
                    let n = d.index(i, k, nz);
                    let xyz = m.mesh.cells[i].center;
                    let mut p = primitive(u[n], m.background[k]);
                    let amp = if strong { 0.05 } else { 1e-5 };
                    p[0] += amp * xyz[0];
                    p[4] += amp * (xyz[2] + 0.2 * k as f64 / nz as f64);
                    p[1] = amp * 30. * xyz[1];
                    p[5] = amp * 0.02;
                    p[6] = amp * 0.03;
                    u[n] = conserved(p, m.background[k]);
                }
            }
            let r = m.rhs(&d, &u);
            let mut w = RhsWorkspace::new(&m, &d);
            let fast = m.rhs_cached(&d, &u, &mut w);
            assert_eq!(r, fast, "Cache changed the physical operator");
            let again = m.rhs_cached(&d, &u, &mut w);
            assert_eq!(r, again);
        }
    }
}
#[test]
fn stretched_shells_equilibrium_and_interpolation() {
    let grid = VerticalGrid::starship_stretched();
    assert_eq!(grid.centers.len(), 149);
    assert_eq!(grid.widths.iter().sum::<f64>(), 1e6);
    let (k, w) = grid.bracket(300000.).unwrap();
    assert_eq!(
        (1. - w) * grid.centers[k] + w * grid.centers[k + 1],
        300000.
    );
    let mut m = Model::isothermal(1, 2, 1000.);
    m.set_vertical(grid);
    let d = Domain::new(&m.mesh, 0, 1);
    let u = m.initial(&d, false);
    let r = m.rhs(&d, &u);
    assert!(r.iter().flatten().all(|v| v.abs() < 1e-12));
    let mut volume = 0.;
    for i in 0..m.mesh.cells.len() {
        for k in 0..m.background.len() {
            volume += m.volume(i, k);
        }
    }
    let expected = 4. * std::f64::consts::PI * ((RE + 1e6).powi(3) - RE.powi(3)) / 3.;
    assert!((volume / expected - 1.).abs() < 1e-13);
}
#[test]
fn nonuniform_closed_mass_species_flux() {
    let mut m = Model::isothermal(2, 2, 1000.);
    m.set_vertical(VerticalGrid::new(vec![0., 1000., 3000., 7000., 13000.]));
    m.reflect_top = true;
    let d = Domain::new(&m.mesh, 0, 1);
    let mut u = m.initial(&d, false);
    let nz = m.background.len();
    for &i in &d.owned {
        for k in 0..nz {
            let n = d.index(i, k, nz);
            let mut p = primitive(u[n], m.background[k]);
            let v = cross([0., 0., 0.02], m.mesh.cells[i].center);
            p[1] = v[0];
            p[2] = v[1];
            p[3] = v[2];
            p[5] = 0.001;
            p[6] = 0.002;
            u[n] = conserved(p, m.background[k]);
        }
    }
    let r = m.rhs(&d, &u);
    for f in [0, 5, 6] {
        let mut signed = 0.;
        let mut abs = 0.;
        for &i in &d.owned {
            for k in 0..nz {
                let v = r[d.index(i, k, nz)][f] * m.volume(i, k);
                signed += v;
                abs += v.abs();
            }
        }
        assert!(signed.abs() / abs.max(1.) < 1e-10);
    }
}
