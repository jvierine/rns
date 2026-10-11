//! Batched HLLE faces. CPU geometry/reconstruction and accumulation remain f64.
use crate::{mesh::V, physics::*};
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision { F32, F64 }
impl Precision {
    pub fn parse(s: &str) -> Result<Self> {
        match s { "f32" => Ok(Self::F32), "f64" => Ok(Self::F64), _ => anyhow::bail!("precision must be f32 or f64") }
    }
}
#[derive(Clone, Copy)]
pub struct Face {
    pub left: Q, pub right: Q, pub al: Atmos, pub ar: Atmos,
    pub normal: V, pub fraction: f64,
}
impl Face {
    pub fn cpu(&self) -> Q { hll_at(self.left, self.right, self.al, self.ar, self.normal, self.fraction) }
    pub fn pack(&self) -> ([f64; 24], Q) {
        let (l, r, a) = face_states(self.left, self.right, self.al, self.ar, self.fraction);
        let gamma = 1. + a.r/a.cv;
        let c = (gamma*a.p/a.rho).sqrt();
        let e = a.p*a.cv/a.r;
        let scales = [a.rho, a.rho*c, a.rho*c, a.rho*c, e, a.rho, a.rho];
        let mut packed = [0.;24];
        let resting = self.left == base(self.al) && self.right == base(self.ar);
        if !resting {
            for (side, q) in [l,r].iter().enumerate() {
                for f in 0..7 { packed[side*7+f] = (q[f]-base(a)[f])/scales[f]; }
            }
        }
        packed[14..17].copy_from_slice(&self.normal);
        packed[17] = gamma;
        let rch = 1.380649e-23/(16.043*1.66053906660e-27);
        let ro2 = 1.380649e-23/(31.998*1.66053906660e-27);
        packed[18] = rch/a.r; packed[19] = ro2/a.r;
        packed[20] = 3.*rch/a.cv; packed[21] = 2.5*ro2/a.cv;
        if !resting { packed[22]=(thermo(l,a).0-a.p)/a.p;packed[23]=(thermo(r,a).0-a.p)/a.p; }
        (packed, scales.map(|s|s*c))
    }
}

pub struct Accelerator {
    pub label:String,
    pub precision:Precision,
    #[cfg(all(feature="native-metal",target_os="macos"))]
    native:crate::native_metal::NativeMetal,
}
impl Accelerator {
    pub fn discover(backend:&str,precision:Precision,_index:usize)->Result<Option<Self>> {
        ensure!(["auto","cpu","metal","gpu"].contains(&backend),"Backend must be auto, cpu, or metal");
        if backend=="cpu" {return Ok(None);}
        #[cfg(all(feature="native-metal",target_os="macos"))]
        if precision==Precision::F32 {
            if let Ok(native)=crate::native_metal::NativeMetal::new(){
                return Ok(Some(Self{label:"Native Metal / f32".into(),precision,native}));
            }
        }
        Ok(None)
    }
    pub fn model_fluxes(&mut self,m:&crate::solver::Model,d:&crate::parallel::Domain,u:&[Q],columns:&[usize],edges:&[usize])->Result<Vec<Q>> {
        #[cfg(all(feature="native-metal",target_os="macos"))]
        {return self.native.model_fluxes(m,d,u,columns,edges);}
        #[cfg(not(all(feature="native-metal",target_os="macos")))]
        {let _=(m,d,u,columns,edges);anyhow::bail!("Metal disabled")}
    }
    pub fn fluxes(&self,faces:&[Face])->Result<Vec<Q>> {
        #[cfg(all(feature="native-metal",target_os="macos"))]
        {
            if faces.is_empty(){return Ok(vec![]);}
            let pairs=faces.iter().map(Face::pack).collect::<Vec<_>>();
            let input=pairs.iter().flat_map(|(p,_)|p.map(|x|x as f32)).collect::<Vec<_>>();
            let result=self.native.fluxes(&input)?;
            return Ok(result.chunks_exact(7).zip(pairs).map(|(q,(_,s))|std::array::from_fn(|f|q[f]as f64*s[f])).collect());
        }
        #[cfg(not(all(feature="native-metal",target_os="macos")))]
        {let _=faces;anyhow::bail!("Metal disabled")}
    }
}
