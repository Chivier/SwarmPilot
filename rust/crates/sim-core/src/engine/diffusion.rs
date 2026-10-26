//! Diffusion engine: takes up to `max_batch` waiting requests, runs the
//! batch's longest step count, decodes, and finishes them together.

use std::collections::VecDeque;

use super::{EngineModel, Event, EventKind, Load, Request, Work};
use crate::profile::DiffusionProfile;

const REF_PIXELS: f64 = 1024.0 * 1024.0;

/// See the module docs.
pub struct DiffusionEngine {
    p: DiffusionProfile,
    waiting: VecDeque<(u64, u32, f64)>,
    batch: Vec<u64>,
    batch_end: Option<f64>,
    /// When the next batch should form, while none runs.
    start_at: Option<f64>,
    rejected: Vec<Event>,
}

impl DiffusionEngine {
    pub fn new(p: DiffusionProfile) -> Self {
        Self {
            p,
            waiting: VecDeque::new(),
            batch: Vec::new(),
            batch_end: None,
            start_at: None,
            rejected: Vec::new(),
        }
    }

    fn start_batch(&mut self, now: f64) {
        let n = (self.p.max_batch.max(1) as usize).min(self.waiting.len());
        if n == 0 {
            self.batch_end = None;
            return;
        }
        let mut steps = 0u32;
        let mut scale: f64 = 0.0;
        for (id, s, px) in self.waiting.drain(..n) {
            self.batch.push(id);
            steps = steps.max(s);
            scale = scale.max(px);
        }
        let batch_factor = 1.0 + self.p.batch_alpha * (n as f64 - 1.0);
        let step = self.p.ms_per_step_1024 * scale * batch_factor;
        let vae = self.p.vae_ms * scale * n as f64;
        self.batch_end = Some(now + steps as f64 * step + vae);
    }
}

impl EngineModel for DiffusionEngine {
    fn submit(&mut self, now: f64, req: Request) {
        let Work::Image {
            steps,
            width,
            height,
        } = req.work
        else {
            self.rejected.push(Event {
                id: req.id,
                t_ms: now,
                kind: EventKind::Rejected,
            });
            return;
        };
        let px = (width.max(1) as f64 * height.max(1) as f64) / REF_PIXELS;
        self.waiting.push_back((req.id, steps.max(1), px));
        if self.batch_end.is_none() && self.start_at.is_none() {
            // Formed in `advance`, so requests arriving together batch.
            self.start_at = Some(now);
        }
    }

    fn next_event(&self) -> Option<f64> {
        [
            self.rejected.first().map(|e| e.t_ms),
            self.batch_end,
            self.start_at,
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
    }

    fn advance(&mut self, now: f64, out: &mut Vec<Event>) {
        out.append(&mut self.rejected);
        loop {
            if let Some(t) = self.batch_end.filter(|&t| t <= now) {
                for id in self.batch.drain(..) {
                    out.push(Event {
                        id,
                        t_ms: t,
                        kind: EventKind::Done,
                    });
                }
                self.batch_end = None;
                self.start_at = Some(t);
            } else if let Some(t) = self.start_at.filter(|&t| t <= now) {
                self.start_at = None;
                self.start_batch(t);
            } else {
                break;
            }
        }
    }

    fn load(&self) -> Load {
        Load {
            running: self.batch.len() as u32,
            waiting: self.waiting.len() as u32,
            utilization: self.batch.len() as f64 / self.p.max_batch.max(1) as f64,
        }
    }
}
