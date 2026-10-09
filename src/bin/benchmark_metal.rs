use global_neutral_euler::physics::*;
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let n = std::env::args()
        .nth(1)
        .unwrap_or("1000000".into())
        .parse::<usize>()?;
    let a = Atmos {
        rho: 1e-10,
        p: 1e-4,
        r: 500.,
        cv: 750.,
        g: 9.,
    };
    let faces = (0..n)
        .map(|i| {
            let x = 0.001 * ((i as f64) * 0.01).sin();
            Face {
                left: conserved([1. + x, 10. * x, 20. * x, 0., 1. + x, 0.001, 0.0036], a),
                right: conserved([1. - x, 5. * x, 0., 0., 1. - x, 0.001, 0.0036], a),
                al: a,
                ar: a,
                normal: [1., 0., 0.],
                fraction: 0.5,
            }
        })
        .collect::<Vec<_>>();
    let mut packed = Vec::with_capacity(n * 24 * 4);
    let mut reference = Vec::with_capacity(n * 7 * 8);
    for face in &faces {
        let (p, s) = face.pack();
        for x in p {
            packed.extend_from_slice(&(x as f32).to_le_bytes());
        }
        for (x, scale) in face.cpu().iter().zip(s) {
            reference.extend_from_slice(&(x / scale).to_le_bytes());
        }
    }
    std::fs::write("benchmarks/apple/input.bin", packed)?;
    std::fs::write("benchmarks/apple/reference.bin", reference)?;
    let mut out = vec![[0.; 7]; n];
    let mut times = vec![];
    for _ in 0..10 {
        let start = Instant::now();
        for (dst, face) in out.iter_mut().zip(&faces) {
            *dst = face.cpu();
        }
        std::hint::black_box(&out);
        times.push(start.elapsed().as_secs_f64());
    }
    println!(
        "Rust CPU f64 {} faces {:.6}s",
        n,
        times.iter().sum::<f64>() / times.len() as f64
    );
    let threads = std::thread::available_parallelism()?.get();
    let mut parallel = vec![];
    for _ in 0..10 {
        let start = Instant::now();
        std::thread::scope(|scope| {
            for (dst, src) in out
                .chunks_mut(n.div_ceil(threads))
                .zip(faces.chunks(n.div_ceil(threads)))
            {
                scope.spawn(move || {
                    for (q, face) in dst.iter_mut().zip(src) {
                        *q = face.cpu();
                    }
                });
            }
        });
        std::hint::black_box(&out);
        parallel.push(start.elapsed().as_secs_f64());
    }
    println!(
        "Rust CPU f64 {} threads {:.6}s",
        threads,
        parallel.iter().sum::<f64>() / parallel.len() as f64
    );
    let normalized = faces
        .iter()
        .map(|f| f.pack().0.map(|x| x as f32))
        .collect::<Vec<_>>();
    let mut f32out = vec![[0f32; 7]; n];
    let mut f32times = vec![];
    for _ in 0..10 {
        let start = Instant::now();
        std::thread::scope(|scope| {
            for (dst, src) in f32out
                .chunks_mut(n.div_ceil(threads))
                .zip(normalized.chunks(n.div_ceil(threads)))
            {
                scope.spawn(move || {
                    for (q, a) in dst.iter_mut().zip(src) {
                        *q = normalized_hlle(a);
                    }
                });
            }
        });
        std::hint::black_box(&f32out);
        f32times.push(start.elapsed().as_secs_f64());
    }
    println!(
        "Rust CPU f32 normalized {} threads {:.6}s",
        threads,
        f32times.iter().sum::<f64>() / f32times.len() as f64
    );
    let f = hdf5::File::create("benchmarks/apple/cpu_benchmark.h5")?;
    f.new_dataset_builder()
        .with_data(&times)
        .create("cpu_seconds")?;
    f.new_dataset_builder()
        .with_data(&f32times)
        .create("cpu_parallel_f32_seconds")?;
    f.new_dataset_builder()
        .with_data(&parallel)
        .create("cpu_parallel_seconds")?;
    f.new_attr::<u64>()
        .create("cpu_threads")?
        .write_scalar(&(threads as u64))?;
    f.new_attr::<u64>()
        .create("faces")?
        .write_scalar(&(n as u64))?;
    Ok(())
}
fn normalized_hlle(a: &[f32; 24]) -> [f32; 7] {
    let gamma = a[17];
    let gm = gamma - 1.;
    let rl = 1. + a[0];
    let rr = 1. + a[7];
    let gl = a[0] + a[5] * (a[18] - 1.) + a[6] * (a[19] - 1.);
    let gr = a[7] + a[12] * (a[18] - 1.) + a[13] * (a[19] - 1.);
    let vl = a[0] + a[5] * (a[20] - 1.) + a[6] * (a[21] - 1.);
    let vr = a[7] + a[12] * (a[20] - 1.) + a[13] * (a[21] - 1.);
    let pl = a[22];
    let pr = a[23];
    let ul = (a[1] * a[14] + a[2] * a[15] + a[3] * a[16]) / rl;
    let ur = (a[8] * a[14] + a[9] * a[15] + a[10] * a[16]) / rr;
    let cl = ((1. + gm * (1. + gl) / (1. + vl)) * (1. + pl) / (gamma * rl)).sqrt();
    let cr = ((1. + gm * (1. + gr) / (1. + vr)) * (1. + pr) / (gamma * rr)).sqrt();
    let sl = 0f32.min(ul - cl).min(ur - cr);
    let sr = 0f32.max(ul + cl).max(ur + cr);
    std::array::from_fn(|f| {
        let mut fl = a[f] * ul;
        let mut fr = a[7 + f] * ur;
        if f == 0 {
            fl = rl * ul;
            fr = rr * ur;
        }
        if f > 0 && f < 4 {
            fl += pl * a[13 + f] / gamma;
            fr += pr * a[13 + f] / gamma;
        }
        if f == 4 {
            fl = (1. + a[4] + gm * (1. + pl)) * ul;
            fr = (1. + a[11] + gm * (1. + pr)) * ur;
        }
        (sr * fl - sl * fr + sl * sr * (a[7 + f] - a[f])) / (sr - sl)
    })
}

#[derive(Clone, Copy)]
pub struct Face {
    pub left: Q,
    pub right: Q,
    pub al: Atmos,
    pub ar: Atmos,
    pub normal: [f64; 3],
    pub fraction: f64,
}
impl Face {
    pub fn cpu(&self) -> Q {
        hll_at(
            self.left,
            self.right,
            self.al,
            self.ar,
            self.normal,
            self.fraction,
        )
    }
    pub fn pack(&self) -> ([f64; 24], Q) {
        let (l, r, a) = face_states(self.left, self.right, self.al, self.ar, self.fraction);
        let gamma = 1. + a.r / a.cv;
        let c = (gamma * a.p / a.rho).sqrt();
        let e = a.p * a.cv / a.r;
        let scales = [a.rho, a.rho * c, a.rho * c, a.rho * c, e, a.rho, a.rho];
        let mut packed = [0.; 24];
        let resting = self.left == base(self.al) && self.right == base(self.ar);
        if !resting {
            for (side, q) in [l, r].iter().enumerate() {
                for f in 0..7 {
                    packed[side * 7 + f] = (q[f] - base(a)[f]) / scales[f];
                }
            }
        }
        packed[14..17].copy_from_slice(&self.normal);
        packed[17] = gamma;
        let rch = 1.380649e-23 / (16.043 * 1.66053906660e-27);
        let ro2 = 1.380649e-23 / (31.998 * 1.66053906660e-27);
        packed[18] = rch / a.r;
        packed[19] = ro2 / a.r;
        packed[20] = 3. * rch / a.cv;
        packed[21] = 2.5 * ro2 / a.cv;
        if !resting {
            packed[22] = (thermo(l, a).0 - a.p) / a.p;
            packed[23] = (thermo(r, a).0 - a.p) / a.p;
        }
        (packed, scales.map(|s| s * c))
    }
}
