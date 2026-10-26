//! Open-loop driver for one engine: feed it a request stream on a virtual
//! clock and record what happened. The multi-instance discrete-event sim
//! (phase S2) builds on the same engine interface.

use std::collections::HashMap;

use crate::engine::{EngineModel, Event, EventKind, Work};
use crate::metrics::RequestRecord;
use crate::workload::Arrival;

/// Run `arrivals` (sorted by time) through `engine` until it is idle.
pub fn run_single(engine: &mut dyn EngineModel, arrivals: &[Arrival]) -> Vec<RequestRecord> {
    let mut records: Vec<RequestRecord> = arrivals
        .iter()
        .map(|a| RequestRecord {
            id: a.req.id,
            arrival_ms: a.t_ms,
            first_token_ms: None,
            done_ms: None,
            output_tokens: match a.req.work {
                Work::Llm { output_tokens, .. } => output_tokens.max(1),
                _ => 1,
            },
            rejected: false,
        })
        .collect();
    let index: HashMap<u64, usize> = records.iter().enumerate().map(|(i, r)| (r.id, i)).collect();
    let mut events: Vec<Event> = Vec::new();
    let mut next = 0;
    loop {
        let t_arr = arrivals.get(next).map(|a| a.t_ms);
        let t_eng = engine.next_event();
        let now = match (t_arr, t_eng) {
            (None, None) => break,
            (Some(a), None) => a,
            (None, Some(e)) => e,
            (Some(a), Some(e)) => a.min(e),
        };
        // Finish engine work due by `now` before new arrivals join.
        engine.advance(now, &mut events);
        while let Some(a) = arrivals.get(next) {
            if a.t_ms > now {
                break;
            }
            engine.submit(now, a.req);
            next += 1;
        }
        engine.advance(now, &mut events);
        for e in events.drain(..) {
            let Some(&i) = index.get(&e.id) else { continue };
            let r = &mut records[i];
            match e.kind {
                EventKind::FirstToken => r.first_token_ms = Some(e.t_ms),
                EventKind::Done => r.done_ms = Some(e.t_ms),
                EventKind::Rejected => r.rejected = true,
                EventKind::Token => {}
            }
        }
    }
    records
}
