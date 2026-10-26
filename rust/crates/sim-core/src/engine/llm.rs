//! Iteration-level continuous batching, after vLLM's scheduler.
//!
//! Each step runs one decode token for every sequence past prefill, then
//! fills the rest of the token budget with prefill chunks, oldest first.
//! KV-cache blocks are taken as tokens are computed. When a decoding
//! sequence needs a block and none is free, the most recently admitted
//! sequence is preempted and recomputed later, as vLLM's default
//! recompute preemption does.
//!
//! A request reaches the scheduler `request_overhead_ms` after it arrives,
//! standing in for the API server (HTTP, tokenization, hand-off to the
//! engine core); that delay shows up in TTFT and end-to-end time.

use std::collections::VecDeque;

use super::{EngineModel, Event, EventKind, Load, Request, Work};
use crate::profile::LlmProfile;

#[derive(Debug, Clone)]
struct Seq {
    id: u64,
    prompt: u32,
    output: u32,
    /// Tokens to (re)compute before decoding: the prompt, plus tokens
    /// already generated after a preemption.
    prefill_target: u32,
    computed: u32,
    generated: u32,
    blocks: u64,
    /// This step's prefill chunk.
    chunk: u32,
    decoding: bool,
}

impl Seq {
    fn context(&self) -> u64 {
        (self.prompt + self.generated) as u64
    }
}

/// See the module docs.
pub struct LlmEngine {
    p: LlmProfile,
    emit_tokens: bool,
    total_blocks: u64,
    free_blocks: u64,
    /// Requests still in the API server, with the time they reach the
    /// scheduler.
    incoming: VecDeque<(f64, Seq)>,
    waiting: VecDeque<Seq>,
    /// In admission order.
    running: Vec<Seq>,
    step_end: Option<f64>,
    /// When the next step should be planned, while no step runs.
    start_at: Option<f64>,
    rejected: Vec<Event>,
}

impl LlmEngine {
    pub fn new(p: LlmProfile, emit_tokens: bool) -> Self {
        let total_blocks = (p.kv_capacity_tokens / p.block_size as u64).max(1);
        Self {
            p,
            emit_tokens,
            total_blocks,
            free_blocks: total_blocks,
            incoming: VecDeque::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            step_end: None,
            start_at: None,
            rejected: Vec::new(),
        }
    }

    fn blocks_for(&self, tokens: u64) -> u64 {
        tokens.div_ceil(self.p.block_size as u64)
    }

    fn preempt_newest(&mut self) -> bool {
        let Some(mut s) = self.running.pop() else {
            return false;
        };
        self.free_blocks += s.blocks;
        s.blocks = 0;
        s.computed = 0;
        s.chunk = 0;
        s.decoding = false;
        s.prefill_target = s.prompt + s.generated;
        self.waiting.push_front(s);
        true
    }

    /// Plan the next step from `now`. Leaves `step_end` as `None` when there
    /// is nothing to do.
    fn start_step(&mut self, now: f64) {
        let mut budget = self.p.max_num_batched_tokens as u64;

        // 1. Decodes, oldest first; make room by preempting the newest.
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].computed < self.running[i].prefill_target {
                i += 1;
                continue;
            }
            // The step writes KV for the newest token, so the cache must
            // hold the whole context (the last token is not cached yet).
            let need = self.blocks_for(self.running[i].context());
            let extra = need.saturating_sub(self.running[i].blocks);
            while extra > self.free_blocks {
                let last = self.running.len() - 1;
                if last == i {
                    break;
                }
                self.preempt_newest();
            }
            if extra > self.free_blocks {
                // Only this sequence is left and it cannot grow.
                self.preempt_newest();
                break;
            }
            self.free_blocks -= extra;
            let s = &mut self.running[i];
            s.blocks += extra;
            s.decoding = true;
            s.chunk = 0;
            budget = budget.saturating_sub(1);
            i += 1;
        }

        // 2. Unfinished prefills already running.
        for s in self.running.iter_mut() {
            if s.decoding || s.computed >= s.prefill_target {
                continue;
            }
            let bs = self.p.block_size as u64;
            let want = (s.prefill_target - s.computed) as u64;
            let room = ((s.blocks + self.free_blocks) * bs).saturating_sub(s.computed as u64);
            let take = want.min(budget).min(room);
            let extra = (s.computed as u64 + take)
                .div_ceil(bs)
                .saturating_sub(s.blocks);
            self.free_blocks -= extra;
            s.blocks += extra;
            s.chunk = take as u32;
            budget -= take;
        }

        // 3. Admit waiting sequences, first come first served.
        while budget > 0 && (self.running.len() as u32) < self.p.max_num_seqs {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let want = front.prefill_target as u64;
            let take = want.min(budget);
            let need = self.blocks_for(take);
            if need > self.free_blocks {
                break;
            }
            let mut s = self.waiting.pop_front().unwrap();
            self.free_blocks -= need;
            s.blocks = need;
            s.chunk = take as u32;
            budget -= take;
            self.running.push(s);
        }

        let mut decode_n = 0u64;
        let mut ctx = 0u64;
        let mut prefill = 0u64;
        for s in &self.running {
            if s.decoding {
                decode_n += 1;
                ctx += s.context();
            } else {
                prefill += s.chunk as u64;
                // Attention over the chunk reads the prefix computed so far.
                ctx += s.computed as u64;
            }
        }
        if decode_n == 0 && prefill == 0 {
            self.step_end = None;
            return;
        }
        let memory = self.p.weight_ms + ctx as f64 * self.p.kv_ms_per_token;
        let compute = (decode_n + prefill) as f64 * self.p.compute_ms_per_token;
        self.step_end = Some(now + self.p.step_overhead_ms + memory.max(compute));
    }

    fn finish_step(&mut self, t: f64, out: &mut Vec<Event>) {
        let mut keep = Vec::with_capacity(self.running.len());
        for mut s in self.running.drain(..) {
            let mut emitted = false;
            if s.decoding {
                s.generated += 1;
                emitted = true;
            } else if s.chunk > 0 {
                s.computed += s.chunk;
                if s.computed >= s.prefill_target {
                    s.generated += 1;
                    emitted = true;
                }
            }
            s.chunk = 0;
            s.decoding = false;
            if emitted {
                if s.generated == 1 {
                    out.push(Event {
                        id: s.id,
                        t_ms: t,
                        kind: EventKind::FirstToken,
                    });
                } else if self.emit_tokens {
                    out.push(Event {
                        id: s.id,
                        t_ms: t,
                        kind: EventKind::Token,
                    });
                }
            }
            if s.generated >= s.output {
                out.push(Event {
                    id: s.id,
                    t_ms: t,
                    kind: EventKind::Done,
                });
                self.free_blocks += s.blocks;
                continue;
            }
            // After a recompute the prefix now covers the generated tokens.
            if s.computed >= s.prefill_target {
                s.computed = s.prompt + s.generated;
                s.prefill_target = s.computed;
            }
            keep.push(s);
        }
        self.running = keep;
    }
}

impl EngineModel for LlmEngine {
    fn submit(&mut self, now: f64, req: Request) {
        let Work::Llm {
            prompt_tokens,
            output_tokens,
        } = req.work
        else {
            self.rejected.push(Event {
                id: req.id,
                t_ms: now,
                kind: EventKind::Rejected,
            });
            return;
        };
        let prompt = prompt_tokens.max(1);
        // A sequence that can never fit in the cache would stall the queue.
        if self.blocks_for((prompt + output_tokens) as u64) > self.total_blocks {
            self.rejected.push(Event {
                id: req.id,
                t_ms: now,
                kind: EventKind::Rejected,
            });
            return;
        }
        let seq = Seq {
            id: req.id,
            prompt,
            output: output_tokens.max(1),
            prefill_target: prompt,
            computed: 0,
            generated: 0,
            blocks: 0,
            chunk: 0,
            decoding: false,
        };
        self.incoming
            .push_back((now + self.p.request_overhead_ms, seq));
    }

    fn next_event(&self) -> Option<f64> {
        [
            self.rejected.first().map(|e| e.t_ms),
            self.incoming.front().map(|x| x.0),
            self.step_end,
            self.start_at,
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
    }

    fn advance(&mut self, now: f64, out: &mut Vec<Event>) {
        out.append(&mut self.rejected);
        // At equal times: a step ends, then arrivals join the queue, then
        // the next step is planned, so simultaneous arrivals batch together.
        loop {
            let due = |t: Option<f64>| t.filter(|&t| t <= now);
            let ts = due(self.step_end);
            let ti = due(self.incoming.front().map(|x| x.0));
            let tk = due(self.start_at);
            let first = [ts, ti, tk].into_iter().flatten().reduce(f64::min);
            let Some(t) = first else { break };
            if ts == Some(t) {
                self.finish_step(t, out);
                self.step_end = None;
                self.start_at = Some(t);
            } else if ti == Some(t) {
                let (_, seq) = self.incoming.pop_front().unwrap();
                self.waiting.push_back(seq);
                if self.step_end.is_none() && self.start_at.is_none() {
                    self.start_at = Some(t);
                }
            } else {
                self.start_at = None;
                self.start_step(t);
            }
        }
    }

    fn load(&self) -> Load {
        Load {
            running: self.running.len() as u32,
            waiting: (self.waiting.len() + self.incoming.len()) as u32,
            utilization: 1.0 - self.free_blocks as f64 / self.total_blocks as f64,
        }
    }
}
