//! The routing state the Scheduler sends, in the form the gateway uses.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use serde::Deserialize;
use swarmpilot_core::mlp::RuntimePredictor;
use swarmpilot_core::{Extractor, StrategyKind};

use crate::pb;

/// Paths that skip the queue when the engine's adapter does not declare
/// them (`PRIORITY_PATHS` of the Python proxy).
pub const PRIORITY_PATHS: [&str; 4] = ["v1/models", "health", "ping", "version"];

/// One instance as the gateway routes to it.
#[derive(Clone, Debug)]
pub struct Instance {
    pub id: String,
    pub endpoint: String,
    pub model_id: String,
    pub software: String,
    /// `json.dumps(platform_info, sort_keys=True)`.
    pub platform_key: String,
    pub active: bool,
    pub max_in_flight: usize,
    pub servers: u32,
    pub extra_headers: Vec<(String, String)>,
}

/// A public route of an engine adapter.
#[derive(Clone, Debug, Deserialize)]
pub struct Route {
    pub path: String,
    #[serde(default)]
    pub lightweight: bool,
    #[serde(default)]
    pub retry_safe: bool,
}

#[derive(Deserialize)]
struct FeatureExpr {
    name: String,
    expr: String,
}

#[derive(Deserialize)]
struct Workload {
    feature_names: Vec<String>,
}

#[derive(Deserialize)]
struct Manifest {
    name: String,
    workload: Workload,
    routes: Vec<Route>,
    features: Option<Vec<FeatureExpr>>,
    #[serde(default)]
    native_styles: Vec<String>,
    #[serde(default = "default_max_tokens")]
    default_max_tokens: i64,
}

fn default_max_tokens() -> i64 {
    4096
}

/// How the gateway computes an adapter's predictor features.
pub enum Features {
    /// Declarative extractors from the manifest.
    Extractors(Vec<(String, Extractor)>),
    /// The pass-through adapter's Python `features()`, ported.
    Passthrough,
    /// Python-only feature code: no features (the request is routed
    /// without a prediction).
    None,
}

/// An engine adapter manifest, compiled.
pub struct Adapter {
    pub name: String,
    pub routes: Vec<Route>,
    pub features: Features,
    pub feature_names: Vec<String>,
    pub native_styles: Vec<String>,
    /// Output limit for translated requests to engine styles that need one.
    pub default_max_tokens: i64,
}

impl Adapter {
    pub fn parse(json: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let features = match m.features {
            Some(list) => Features::Extractors(
                list.into_iter()
                    .map(|f| {
                        Extractor::compile(&f.expr, 0.0)
                            .map(|x| (f.name, x))
                            .map_err(|e| e.0)
                    })
                    .collect::<Result<_, _>>()?,
            ),
            None if m.name == "passthrough" => Features::Passthrough,
            None => Features::None,
        };
        Ok(Adapter {
            name: m.name,
            routes: m.routes,
            features,
            feature_names: m.workload.feature_names,
            native_styles: m.native_styles,
            default_max_tokens: m.default_max_tokens,
        })
    }

    pub fn route(&self, path: &str) -> Option<&Route> {
        let path = path.trim_start_matches('/');
        self.routes.iter().find(|r| r.path == path)
    }

    /// `_is_lightweight` of the Python proxy.
    pub fn is_lightweight(&self, path: &str) -> bool {
        match self.route(path) {
            Some(r) => r.lightweight,
            None => PRIORITY_PATHS.contains(&path),
        }
    }
}

/// Trained models and hardware features of one platform.
pub struct Platform {
    pub quantile_key: String,
    pub expect_error_key: String,
    pub hardware_features: HashMap<String, f64>,
    pub experiment: bool,
}

/// A loaded portable model and the sidecar time it was loaded at.
struct Loaded {
    modified: SystemTime,
    model: Arc<RuntimePredictor>,
}

/// Portable models from the predictor storage directory, reloaded when
/// their sidecar changes (a retrain writes a new one).
#[derive(Default)]
pub struct ModelCache {
    dir: PathBuf,
    loaded: HashMap<String, Loaded>,
}

impl ModelCache {
    pub fn set_dir(&mut self, dir: &str) {
        let dir = PathBuf::from(dir);
        if dir != self.dir {
            self.dir = dir;
            self.loaded.clear();
        }
    }

    /// The model stored under `key`, if its files exist and load.
    pub fn get(&self, key: &str) -> Option<Arc<RuntimePredictor>> {
        self.loaded.get(key).map(|l| l.model.clone())
    }

    /// Load new or changed models for `keys`; forget deleted ones.
    pub fn refresh<'a>(&mut self, keys: impl Iterator<Item = &'a str>) {
        let mut seen = Vec::new();
        for key in keys {
            if key.is_empty() || key.contains('/') || key.contains('\\') {
                continue;
            }
            seen.push(key.to_string());
            let sidecar = self.dir.join(format!("{key}.json"));
            let modified = match std::fs::metadata(&sidecar).and_then(|m| m.modified()) {
                Ok(t) => t,
                Err(_) => {
                    self.loaded.remove(key);
                    continue;
                }
            };
            if self.loaded.get(key).is_some_and(|l| l.modified == modified) {
                continue;
            }
            match load(&sidecar) {
                Ok(model) => {
                    tracing::info!(key, "loaded predictor model");
                    self.loaded.insert(
                        key.to_string(),
                        Loaded {
                            modified,
                            model: Arc::new(model),
                        },
                    );
                }
                Err(e) => {
                    // A model being written right now; the next refresh
                    // picks it up.
                    tracing::warn!(key, error = %e, "cannot load predictor model");
                    self.loaded.remove(key);
                }
            }
        }
        self.loaded.retain(|k, _| seen.contains(k));
    }
}

fn load(sidecar: &Path) -> Result<RuntimePredictor, String> {
    RuntimePredictor::load_portable(sidecar).map_err(|e| e.to_string())
}

/// Everything routing needs from one Scheduler state.
pub struct Snapshot {
    pub version: u64,
    pub model_id: String,
    pub strategy: StrategyKind,
    pub strategy_name: String,
    pub timeout_s: f64,
    pub http_timeout_s: f64,
    pub max_retries: u32,
    pub retry_delay_s: f64,
    /// All registered instances, in registry order.
    pub instances: Vec<Instance>,
    pub adapters: HashMap<String, Arc<Adapter>>,
    pub passthrough: Arc<Adapter>,
    pub platforms: HashMap<String, Platform>,
    pub sketch_quantiles: Vec<f64>,
    pub preprocessor_rules: bool,
    pub normals: Arc<Vec<f64>>,
    pub style_paths: HashMap<String, String>,
    /// Refuse answer-changing fields that translation would drop.
    pub translation_strict: bool,
    pub storage_dir: String,
}

impl Snapshot {
    pub fn from_proto(s: pb::State) -> Result<Self, String> {
        let mut adapters = HashMap::new();
        for (software, json) in &s.manifests {
            let adapter = Adapter::parse(json).map_err(|e| format!("manifest {software}: {e}"))?;
            if matches!(adapter.features, Features::None) {
                tracing::warn!(
                    engine = %software,
                    adapter = %adapter.name,
                    "adapter computes features in Python only; its requests are routed without predictions"
                );
            }
            adapters.insert(software.clone(), Arc::new(adapter));
        }
        let passthrough = adapters
            .get("")
            .cloned()
            .ok_or("state has no pass-through manifest")?;
        let proxy = s.proxy.unwrap_or_default();
        let predictor = s.predictor.unwrap_or_default();
        if predictor.preprocessor_rules {
            tracing::warn!(
                "a preprocessor chain is configured for this model; the gateway cannot run it and routes without trained predictions"
            );
        }
        let instances = s
            .instances
            .into_iter()
            .map(|i| {
                let software = i
                    .platform_info
                    .get("software_name")
                    .cloned()
                    .unwrap_or_default();
                let mut extra: Vec<(String, String)> = i.extra_headers.into_iter().collect();
                extra.sort();
                Instance {
                    platform_key: i.platform_key,
                    active: i.status == pb::instance::Status::Active as i32,
                    max_in_flight: i.max_in_flight.max(1) as usize,
                    servers: i.servers.max(1),
                    id: i.instance_id,
                    endpoint: i.endpoint,
                    model_id: i.model_id,
                    software,
                    extra_headers: extra,
                }
            })
            .collect();
        let platforms = predictor
            .platforms
            .into_iter()
            .map(|p| {
                (
                    p.platform_key,
                    Platform {
                        quantile_key: p.quantile_model_key,
                        expect_error_key: p.expect_error_model_key,
                        hardware_features: p.hardware_features,
                        experiment: p.experiment,
                    },
                )
            })
            .collect();
        Ok(Snapshot {
            version: s.version,
            model_id: s.model_id,
            strategy: StrategyKind::from_name(&s.strategy),
            strategy_name: s.strategy,
            timeout_s: if proxy.timeout_s > 0.0 {
                proxy.timeout_s
            } else {
                300.0
            },
            http_timeout_s: if proxy.http_timeout_s > 0.0 {
                proxy.http_timeout_s
            } else {
                300.0
            },
            max_retries: proxy.max_retries.max(1),
            retry_delay_s: proxy.retry_delay_s.max(0.0),
            instances,
            adapters,
            passthrough,
            platforms,
            sketch_quantiles: predictor.sketch_quantiles,
            preprocessor_rules: predictor.preprocessor_rules,
            normals: Arc::new(predictor.experiment_normals),
            style_paths: s.api_style_paths,
            translation_strict: !proxy.translation_lenient,
            storage_dir: predictor.storage_dir,
        })
    }

    /// `resolve_adapter(software_name)`.
    pub fn adapter(&self, software: &str) -> Arc<Adapter> {
        self.adapters
            .get(software)
            .cloned()
            .unwrap_or_else(|| self.passthrough.clone())
    }

    pub fn instance(&self, id: &str) -> Option<&Instance> {
        self.instances.iter().find(|i| i.id == id)
    }
}
