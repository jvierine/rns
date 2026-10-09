//! Physical altitude faces. Every geometric and source operation uses this mesh.
#[derive(Clone, Debug)]
pub struct VerticalGrid {
    pub faces: Vec<f64>,
    pub centers: Vec<f64>,
    pub widths: Vec<f64>,
}
impl VerticalGrid {
    pub fn new(faces: Vec<f64>) -> Self {
        assert!(faces.len() >= 3 && faces[0] == 0.);
        assert!(faces.iter().all(|x| x.is_finite()));
        assert!(faces.windows(2).all(|w| w[1] > w[0]));
        let centers = faces.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let widths = faces.windows(2).map(|w| w[1] - w[0]).collect();
        Self {
            faces,
            centers,
            widths,
        }
    }
    pub fn uniform(nz: usize, dz: f64) -> Self {
        Self::new((0..=nz).map(|k| k as f64 * dz).collect())
    }
    pub fn starship_stretched() -> Self {
        let mut faces = vec![0.];
        for (end, step) in [
            (150_000., 2500.),
            (400_000., 5000.),
            (650_000., 10_000.),
            (1_000_000., 25_000.),
        ] {
            while *faces.last().unwrap() < end {
                faces.push(faces.last().unwrap() + step);
            }
        }
        Self::new(faces)
    }
    /// Centre interpolation with no invented support outside the centre range.
    pub fn bracket(&self, height: f64) -> Option<(usize, f64)> {
        if !height.is_finite() || height < self.centers[0] || height > *self.centers.last().unwrap()
        {
            return None;
        }
        let k = self
            .centers
            .partition_point(|x| *x <= height)
            .saturating_sub(1)
            .min(self.centers.len() - 2);
        Some((
            k,
            (height - self.centers[k]) / (self.centers[k + 1] - self.centers[k]),
        ))
    }
    pub fn uniform_spacing(&self) -> Option<f64> {
        let dz = self.widths[0];
        self.widths
            .iter()
            .all(|w| (*w - dz).abs() < 1e-9)
            .then_some(dz)
    }
}
