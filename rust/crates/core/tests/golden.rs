//! Replay the golden vectors of `tests/golden/` (recorded from the Python
//! Scheduler) against the Rust core. Decisions must match exactly and
//! every number to the last bit, except where noted.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use serde_json::Value;
use swarmpilot_core::sketch::{Charge, Ledger, Sketch, SketchKind};
use swarmpilot_core::strategy::{Draws, Prediction, Router, StrategyKind};
use swarmpilot_core::Extractor;

fn golden(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/golden")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn pairs(v: &Value) -> Option<Vec<(f64, f64)>> {
    v.as_array()
        .map(|a| a.iter().map(|p| (f(&p[0]), f(&p[1]))).collect::<Vec<_>>())
}

/// A sketch as the golden files write it.
fn sketch_json(s: &Sketch) -> Value {
    match s {
        Sketch::Probabilistic { quantiles, values } => {
            serde_json::json!({"quantiles": quantiles, "values": values})
        }
        Sketch::ExpectError {
            expected_time_ms,
            error_margin_ms,
        } => serde_json::json!({
            "expected_time_ms": expected_time_ms,
            "error_margin_ms": error_margin_ms,
        }),
    }
}

/// Compare a sketch with its golden record, bit for bit.
fn assert_sketch(ctx: &str, got: &Sketch, want: &Value) {
    let got = sketch_json(got);
    assert_eq!(&got, want, "{ctx}: sketch differs");
}

#[test]
fn sketches_match_python() {
    let file = golden("sketches.json");
    for case in file["cases"].as_array().unwrap() {
        let kind = SketchKind::from_name(case["kind"].as_str().unwrap());
        let mut ledger = Ledger::new(kind, &floats(&case["quantile_grid"]));
        ledger.register("i0").unwrap();
        ledger.register("i1").unwrap();
        let mut open: Vec<String> = Vec::new();
        for (n, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let task = step["task_id"].as_str().unwrap_or_default();
            match step["op"].as_str().unwrap() {
                "charge" => {
                    let c = &step["charge"];
                    ledger.charge(
                        Some(task),
                        Charge::of(
                            c["instance_id"].as_str().unwrap(),
                            f(&c["time_ms"]),
                            c["error_ms"].as_f64(),
                            pairs(&c["quantiles"]).as_deref(),
                        ),
                    );
                    open.push(task.to_string());
                }
                "release" => {
                    assert!(ledger.release(task).is_some());
                    open.retain(|t| t != task);
                }
                "move" => {
                    assert!(ledger.move_charge(task, step["instance_id"].as_str().unwrap()));
                }
                "release_all" => {
                    for t in open.drain(..) {
                        ledger.release(&t);
                    }
                }
                op => panic!("unknown op {op}"),
            }
            for (iid, want) in step["sketches"].as_object().unwrap() {
                let ctx = format!("{} step {n} {iid}", kind.name());
                assert_sketch(&ctx, ledger.sketch(iid).unwrap(), want);
            }
        }
    }
}

/// Replays the draws Python's generator made.
struct Replay(VecDeque<Value>);

impl Replay {
    fn next(&mut self, op: &str) -> Value {
        let d = self
            .0
            .pop_front()
            .unwrap_or_else(|| panic!("no {op} draw left"));
        assert_eq!(d["op"], op, "draw kind");
        d
    }
}

impl Draws for Replay {
    fn random(&mut self, n: usize) -> Vec<f64> {
        let d = self.next("random");
        assert_eq!(d["size"].as_u64(), Some(n as u64));
        floats(&d["out"])
    }
    fn integers(&mut self, high: usize) -> usize {
        let d = self.next("integers");
        assert_eq!(d["high"].as_u64(), Some(high as u64));
        d["out"].as_u64().unwrap() as usize
    }
    fn choice2(&mut self, n: usize) -> (usize, usize) {
        let d = self.next("choice");
        assert_eq!(d["n"].as_u64(), Some(n as u64));
        let out = d["out"].as_array().unwrap();
        (
            out[0].as_u64().unwrap() as usize,
            out[1].as_u64().unwrap() as usize,
        )
    }
}

#[test]
fn strategies_match_python() {
    let file = golden("strategies.json");
    let grid = floats(&file["quantile_grid"]);
    let runs = file["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 7);
    for run in runs {
        let name = run["strategy"].as_str().unwrap();
        let kind = StrategyKind::from_name(name);
        assert_eq!(kind.sketch_kind().name(), run["queue_kind"]);
        let mut ledger = Ledger::new(kind.sketch_kind(), &grid);
        let warm_hw: HashSet<&str> = run["warm_hardware"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut ids = Vec::new();
        let mut warm = Vec::new();
        for inst in run["instances"].as_array().unwrap() {
            let id = inst["instance_id"].as_str().unwrap().to_string();
            ledger.register(&id).unwrap();
            warm.push(warm_hw.contains(inst["platform_info"]["hardware_name"].as_str().unwrap()));
            ids.push(id);
        }
        let mut router = Router::new(kind);
        for (n, step) in run["steps"].as_array().unwrap().iter().enumerate() {
            let ctx = format!("{name} step {n}");
            let task = step["task_id"].as_str().unwrap();
            if step["op"] == "finish" {
                let released = ledger.release(task);
                assert_eq!(
                    released.as_ref().map(|c| c.instance_id.as_str()),
                    step["instance_id"].as_str(),
                    "{ctx}"
                );
                match released {
                    Some(c) => assert_sketch(
                        &ctx,
                        ledger.sketch(&c.instance_id).unwrap(),
                        &step["sketch"],
                    ),
                    None => assert!(step["sketch"].is_null(), "{ctx}"),
                }
                continue;
            }
            let table = &step["predictions"];
            let mut predict = |ptype: &'static str, subset: &[usize]| -> Vec<Prediction> {
                subset
                    .iter()
                    .map(|&k| {
                        let row = &table[&ids[k]];
                        Prediction {
                            instance_id: ids[k].clone(),
                            predicted_time_ms: f(&row["predicted_time_ms"]),
                            error_margin_ms: (ptype == "expect_error")
                                .then(|| f(&row["error_margin_ms"])),
                            quantiles: if ptype == "quantile" {
                                pairs(&row["quantiles"])
                            } else {
                                None
                            },
                        }
                    })
                    .collect()
            };
            let mut draws = Replay(step["draws"].as_array().unwrap().iter().cloned().collect());
            let routed = router.route(
                &mut ledger,
                &ids,
                &warm,
                &mut predict,
                &mut draws,
                Some(task),
            );
            assert!(draws.0.is_empty(), "{ctx}: draws left over");
            assert_eq!(ids[routed.selected], step["selected"], "{ctx}: selection");
            let types: Vec<Value> = routed
                .prediction_types
                .iter()
                .map(|t| Value::from(*t))
                .collect();
            assert_eq!(Value::from(types), step["prediction_types"], "{ctx}");
            let charged = routed.prediction.as_ref().map(|p| {
                serde_json::json!({
                    "time_ms": p.predicted_time_ms,
                    "error_ms": p.error_margin_ms,
                    "has_quantiles": p.quantiles.as_ref().is_some_and(|q| !q.is_empty()),
                })
            });
            assert_eq!(
                charged.unwrap_or(Value::Null),
                step["charged"],
                "{ctx}: charged"
            );
            assert_sketch(
                &ctx,
                ledger.sketch(&ids[routed.selected]).unwrap(),
                &step["sketch"],
            );
        }
    }
}

#[test]
fn features_match_python() {
    let file = golden("features.json");
    let mut checked = 0;
    for adapter in file["adapters"].as_array().unwrap() {
        let Some(exprs) = adapter["extractors"].as_array() else {
            continue;
        };
        let compiled: Vec<(String, Extractor)> = exprs
            .iter()
            .map(|e| {
                (
                    e["name"].as_str().unwrap().to_string(),
                    Extractor::compile(e["expr"].as_str().unwrap(), 0.0).unwrap(),
                )
            })
            .collect();
        for case in adapter["cases"].as_array().unwrap() {
            for (name, ex) in &compiled {
                let got = ex.eval(&&case["body"]);
                assert_eq!(
                    Value::from(got),
                    case["manifest_features"][name],
                    "{} {name} on {}",
                    adapter["adapter"],
                    case["body"]
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "only {checked} features checked");
}
