use crate::{mesh::Mesh, physics::Q};
use mpi::{point_to_point, traits::*};
use std::collections::{BTreeMap, BTreeSet};
pub struct Domain {
    pub rank: usize,
    pub size: usize,
    pub owned: Vec<usize>,
    pub needed: Vec<usize>,
    pub slots: BTreeMap<usize, usize>,
    pub dense_slots: Vec<usize>,
    pub send: Vec<Vec<usize>>,
    pub receive: Vec<Vec<usize>>,
    pub active: Vec<bool>,
}
pub struct ExchangeWorkspace {
    send: Vec<Vec<f64>>,
    receive: Vec<Vec<f64>>,
}
impl Domain {
    pub fn new(m: &Mesh, rank: usize, size: usize) -> Self {
        Self::selected(m, rank, size, &(0..m.cells.len()).collect::<Vec<_>>())
    }
    /// Exterior two-ring columns are fixed ambient ghosts, never rank-owned.
    pub fn selected(m: &Mesh, rank: usize, size: usize, active: &[usize]) -> Self {
        let nc = m.cells.len();
        assert!(size <= active.len());
        let ranges = (0..size)
            .map(|r| active[active.len() * r / size..active.len() * (r + 1) / size].to_vec())
            .collect::<Vec<_>>();
        let needs = ranges
            .iter()
            .map(|range| {
                let mut set = range.iter().copied().collect::<BTreeSet<_>>();
                for _ in 0..2 {
                    let old = set.clone();
                    for i in old {
                        set.extend(m.neighbors(i));
                    }
                }
                set
            })
            .collect::<Vec<_>>();
        let owned = ranges[rank].clone();
        let needed = needs[rank].iter().copied().collect::<Vec<_>>();
        let slots = needed.iter().enumerate().map(|(s, &i)| (i, s)).collect();
        let mut dense_slots = vec![usize::MAX; nc];
        for (s, &i) in needed.iter().enumerate() {
            dense_slots[i] = s;
        }
        let send = (0..size)
            .map(|r| {
                if r == rank {
                    vec![]
                } else {
                    owned
                        .iter()
                        .copied()
                        .filter(|i| needs[r].contains(i))
                        .collect()
                }
            })
            .collect();
        let receive = (0..size)
            .map(|r| {
                if r == rank {
                    vec![]
                } else {
                    ranges[r]
                        .iter()
                        .copied()
                        .filter(|i| needs[rank].contains(i))
                        .collect()
                }
            })
            .collect();
        Self {
            rank,
            size,
            owned,
            needed,
            slots,
            dense_slots,
            send,
            receive,
            active: (0..nc).map(|i| active.binary_search(&i).is_ok()).collect(),
        }
    }
    pub fn index(&self, i: usize, k: usize, nz: usize) -> usize {
        let slot = self.dense_slots[i];
        assert!(slot != usize::MAX, "Missing halo column");
        slot * nz + k
    }
    pub fn exchange(&self, world: &impl Communicator, u: &mut [Q], nz: usize) {
        let mut work = self.exchange_workspace(nz);
        self.exchange_cached(world, u, nz, &mut work);
    }
    pub fn exchange_workspace(&self, nz: usize) -> ExchangeWorkspace {
        ExchangeWorkspace {
            send: self
                .send
                .iter()
                .map(|ids| vec![0.; ids.len() * nz * 7])
                .collect(),
            receive: self
                .receive
                .iter()
                .map(|ids| vec![0.; ids.len() * nz * 7])
                .collect(),
        }
    }
    pub fn exchange_cached(
        &self,
        world: &impl Communicator,
        u: &mut [Q],
        nz: usize,
        work: &mut ExchangeWorkspace,
    ) {
        // Deterministic pair exchanges: only requested two-ring halo columns move.
        // This correctness-first schedule is blocking; asynchronous graph exchanges
        // and a better spatial partition are future performance work.
        for r in 0..self.size {
            if r == self.rank || self.send[r].is_empty() && self.receive[r].is_empty() {
                continue;
            }
            let send = &mut work.send[r];
            let mut n = 0;
            for &i in &self.send[r] {
                for k in 0..nz {
                    send[n..n + 7].copy_from_slice(&u[self.index(i, k, nz)]);
                    n += 7;
                }
            }
            let recv = &mut work.receive[r];
            let process = world.process_at_rank(r as i32);
            point_to_point::send_receive_into(&send[..], &process, &mut recv[..], &process);
            let mut n = 0;
            for &i in &self.receive[r] {
                for k in 0..nz {
                    u[self.index(i, k, nz)].copy_from_slice(&recv[n..n + 7]);
                    n += 7;
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reciprocal_halos() {
        let m = Mesh::new(2);
        for size in [1, 2, 3, 8] {
            let domains = (0..size)
                .map(|r| Domain::new(&m, r, size))
                .collect::<Vec<_>>();
            assert_eq!(
                domains.iter().map(|d| d.owned.len()).sum::<usize>(),
                m.cells.len()
            );
            for a in 0..size {
                for b in 0..size {
                    assert_eq!(domains[a].send[b], domains[b].receive[a]);
                }
            }
        }
    }
}
