//! Generic engine: `concurrency` parallel slots, FIFO queue, lognormal
//! service times with the profile's mean and coefficient of variation.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};

use super::{EngineModel, Event, EventKind, Load, Request};
use crate::profile::GenericProfile;

#[derive(Debug, PartialEq)]
struct Slot(f64, u64);

impl Eq for Slot {}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Slot {
    // Min-heap on finish time, then id, so ties pop in a fixed order.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .total_cmp(&self.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// See the module docs.
pub struct GenericEngine {
    p: GenericProfile,
    rng: ChaCha8Rng,
    dist: LogNormal<f64>,
    busy: BinaryHeap<Slot>,
    waiting: VecDeque<u64>,
}

impl GenericEngine {
    pub fn new(p: GenericProfile, seed: u64) -> Self {
        let sigma2 = (1.0 + p.cv * p.cv).ln();
        let mu = p.mean_ms.max(1e-6).ln() - sigma2 / 2.0;
        let dist = LogNormal::new(mu, sigma2.sqrt()).expect("valid lognormal");
        Self {
            p,
            rng: ChaCha8Rng::seed_from_u64(seed),
            dist,
            busy: BinaryHeap::new(),
            waiting: VecDeque::new(),
        }
    }

    fn start(&mut self, now: f64, id: u64) {
        let d = self.dist.sample(&mut self.rng);
        self.busy.push(Slot(now + d, id));
    }
}

impl EngineModel for GenericEngine {
    fn submit(&mut self, now: f64, req: Request) {
        // Any kind of work is accepted: the engine is opaque.
        if (self.busy.len() as u32) < self.p.concurrency.max(1) {
            self.start(now, req.id);
        } else {
            self.waiting.push_back(req.id);
        }
    }

    fn next_event(&self) -> Option<f64> {
        self.busy.peek().map(|s| s.0)
    }

    fn advance(&mut self, now: f64, out: &mut Vec<Event>) {
        while let Some(Slot(t, _)) = self.busy.peek() {
            if *t > now {
                break;
            }
            let Slot(t, id) = self.busy.pop().unwrap();
            out.push(Event {
                id,
                t_ms: t,
                kind: EventKind::Done,
            });
            if let Some(next) = self.waiting.pop_front() {
                self.start(t, next);
            }
        }
    }

    fn load(&self) -> Load {
        Load {
            running: self.busy.len() as u32,
            waiting: self.waiting.len() as u32,
            utilization: self.busy.len() as f64 / self.p.concurrency.max(1) as f64,
        }
    }
}
