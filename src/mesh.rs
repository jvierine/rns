use std::collections::BTreeMap;
pub type V = [f64; 3];
pub fn add(a: V, b: V) -> V {
    std::array::from_fn(|d| a[d] + b[d])
}
pub fn sub(a: V, b: V) -> V {
    std::array::from_fn(|d| a[d] - b[d])
}
pub fn mul(a: V, s: f64) -> V {
    a.map(|x| x * s)
}
pub fn dot(a: V, b: V) -> f64 {
    (0..3).map(|d| a[d] * b[d]).sum()
}
pub fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
pub fn unit(a: V) -> V {
    mul(a, 1. / norm(a))
}
pub fn angle(a: V, b: V) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}
pub const RE: f64 = 6_371_008.8;
#[derive(Clone)]
pub struct Cell {
    pub vertices: [usize; 3],
    pub center: V,
    pub area: f64,
    pub vector_area: V,
    pub edges: [usize; 3],
}
#[derive(Clone)]
pub struct Edge {
    pub left: usize,
    pub right: usize,
    pub outward: V,
    pub angle: f64,
}
pub struct Mesh {
    pub vertices: Vec<V>,
    pub cells: Vec<Cell>,
    pub edges: Vec<Edge>,
}
impl Mesh {
    pub fn new(level: usize) -> Self {
        assert!(
            level <= 8,
            "Refinement guard; memory grows by four each level"
        );
        let t = (1. + 5f64.sqrt()) / 2.;
        let mut vertices = vec![
            [-1., t, 0.],
            [1., t, 0.],
            [-1., -t, 0.],
            [1., -t, 0.],
            [0., -1., t],
            [0., 1., t],
            [0., -1., -t],
            [0., 1., -t],
            [t, 0., -1.],
            [t, 0., 1.],
            [-t, 0., -1.],
            [-t, 0., 1.],
        ]
        .into_iter()
        .map(unit)
        .collect::<Vec<_>>();
        let mut triangles = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..level {
            let mut cache = BTreeMap::new();
            let mut next = Vec::new();
            for [a, b, c] in triangles {
                let mut mid = |i: usize, j: usize| {
                    let key = (i.min(j), i.max(j));
                    *cache.entry(key).or_insert_with(|| {
                        vertices.push(unit(add(vertices[i], vertices[j])));
                        vertices.len() - 1
                    })
                };
                let ab = mid(a, b);
                let bc = mid(b, c);
                let ca = mid(c, a);
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            triangles = next;
        }
        let mut cells = Vec::new();
        for mut v in triangles {
            if dot(vertices[v[0]], cross(vertices[v[1]], vertices[v[2]])) < 0. {
                v.swap(1, 2);
            }
            let [a, b, c] = v.map(|i| vertices[i]);
            let area = 2. * dot(a, cross(b, c)).atan2(1. + dot(a, b) + dot(b, c) + dot(c, a));
            let mut va = [0.; 3];
            for (x, y) in [(a, b), (b, c), (c, a)] {
                va = add(va, mul(unit(cross(x, y)), angle(x, y) / 2.));
            }
            cells.push(Cell {
                vertices: v,
                center: unit(add(add(a, b), c)),
                area,
                vector_area: va,
                edges: [0; 3],
            });
        }
        let mut edges = Vec::<Edge>::new();
        let mut cache = BTreeMap::<(usize, usize), usize>::new();
        for i in 0..cells.len() {
            for d in 0..3 {
                let a = cells[i].vertices[d];
                let b = cells[i].vertices[(d + 1) % 3];
                let key = (a.min(b), a.max(b));
                let e = if let Some(&e) = cache.get(&key) {
                    edges[e].right = i;
                    e
                } else {
                    let e = edges.len();
                    edges.push(Edge {
                        left: i,
                        right: usize::MAX,
                        outward: mul(unit(cross(vertices[a], vertices[b])), -1.),
                        angle: angle(vertices[a], vertices[b]),
                    });
                    cache.insert(key, e);
                    e
                };
                cells[i].edges[d] = e;
            }
        }
        assert!(edges.iter().all(|e| e.right != usize::MAX));
        Self {
            vertices,
            cells,
            edges,
        }
    }
    pub fn neighbors(&self, i: usize) -> [usize; 3] {
        self.cells[i].edges.map(|e| {
            let e = &self.edges[e];
            if e.left == i { e.right } else { e.left }
        })
    }
    pub fn volume(&self, i: usize, k: usize, dz: f64) -> f64 {
        let lo = RE + k as f64 * dz;
        self.cells[i].area * ((lo + dz).powi(3) - lo.powi(3)) / 3.
    }
    pub fn shell_volume(&self, i: usize, lower_m: f64, upper_m: f64) -> f64 {
        let lo = RE + lower_m;
        let hi = RE + upper_m;
        // Factored shell difference avoids subtraction of nearly equal cubes.
        self.cells[i].area * (hi - lo) * (hi * hi + hi * lo + lo * lo) / 3.
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sphere_and_closed_cells() {
        for l in 0..=4 {
            let m = Mesh::new(l);
            assert_eq!(m.cells.len(), 20 * 4usize.pow(l as u32));
            assert_eq!(m.vertices.len() + m.cells.len() - m.edges.len(), 2);
            assert!(
                (m.cells.iter().map(|c| c.area).sum::<f64>() - 4. * std::f64::consts::PI).abs()
                    < 1e-12
            );
            for (i, c) in m.cells.iter().enumerate() {
                let mut s = mul(c.vector_area, 2.);
                for &eid in &c.edges {
                    let e = &m.edges[eid];
                    s = add(
                        s,
                        mul(e.outward, e.angle * if e.left == i { 1. } else { -1. }),
                    );
                }
                assert!(norm(s) < 1e-13);
                for j in m.neighbors(i) {
                    assert!(m.neighbors(j).contains(&i));
                }
            }
        }
    }
}
