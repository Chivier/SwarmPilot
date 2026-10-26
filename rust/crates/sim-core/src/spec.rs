//! Cluster spec: node groups of one GPU type, each running deployments of
//! one engine and model. `plan` places every instance on a node and
//! resolves its engine profile.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::catalog::Catalog;
use crate::profile::EngineProfile;
use crate::Error;

const DEFAULT_SPEC_YAML: &str = include_str!("../../../../swarmpilot/sim/specs/default-256.yaml");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterSpec {
    pub name: String,
    #[serde(default)]
    pub seed: u64,
    pub node_groups: Vec<NodeGroup>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeGroup {
    pub name: String,
    pub gpu: String,
    pub gpus_per_node: u32,
    pub nodes: u32,
    #[serde(default)]
    pub deployments: Vec<Deployment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deployment {
    pub engine: String,
    pub model: String,
    #[serde(default = "one")]
    pub tp: u32,
    pub instances: u32,
}

fn one() -> u32 {
    1
}

/// One placed instance.
#[derive(Debug, Clone, Serialize)]
pub struct InstancePlan {
    pub id: String,
    pub group: String,
    pub node: String,
    pub gpu_indices: Vec<u32>,
    pub profile: EngineProfile,
}

/// Totals for a planned cluster.
#[derive(Debug, Clone, Serialize)]
pub struct PlanSummary {
    pub name: String,
    pub configurations: usize,
    pub nodes: u32,
    pub gpus: u32,
    pub gpus_used: u32,
    pub instances: usize,
    /// Instances per (engine, model, GPU, TP).
    pub per_profile: BTreeMap<String, usize>,
    pub uncalibrated_instances: usize,
}

impl ClusterSpec {
    /// The 9-configuration, 256-instance default from
    /// `swarmpilot/sim/specs/default-256.yaml`.
    pub fn default_256() -> Self {
        Self::from_yaml(DEFAULT_SPEC_YAML).expect("built-in default spec must parse")
    }

    pub fn from_yaml(text: &str) -> Result<Self, Error> {
        Ok(serde_yaml::from_str(text)?)
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }

    /// Place every instance first-fit on its group's nodes (an instance's
    /// `tp` GPUs stay on one node) and resolve its profile.
    pub fn plan(&self, catalog: &Catalog) -> Result<Vec<InstancePlan>, Error> {
        let mut out = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for g in &self.node_groups {
            if !names.insert(g.name.as_str()) {
                return Err(Error::Invalid(format!("duplicate node group {}", g.name)));
            }
            catalog.gpu(&g.gpu)?;
            let mut free: Vec<Vec<u32>> = (0..g.nodes)
                .map(|_| (0..g.gpus_per_node).collect())
                .collect();
            for d in &g.deployments {
                if d.tp == 0 || d.tp > g.gpus_per_node {
                    return Err(Error::Invalid(format!(
                        "group {}: tp {} does not fit a {}-GPU node",
                        g.name, d.tp, g.gpus_per_node
                    )));
                }
                let profile = EngineProfile::resolve(catalog, &d.engine, &d.model, &g.gpu, d.tp)?;
                for _ in 0..d.instances {
                    let Some(n) = free.iter().position(|f| f.len() >= d.tp as usize) else {
                        return Err(Error::Invalid(format!(
                            "group {}: not enough GPUs for {} x {} (tp {})",
                            g.name, d.instances, profile.model, d.tp
                        )));
                    };
                    let gpus: Vec<u32> = free[n].drain(..d.tp as usize).collect();
                    out.push(InstancePlan {
                        id: format!("{}-{}", g.name, out.len()),
                        group: g.name.clone(),
                        node: format!("{}-n{}", g.name, n),
                        gpu_indices: gpus,
                        profile: profile.clone(),
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn summarize(&self, plan: &[InstancePlan]) -> PlanSummary {
        let mut per_profile = BTreeMap::new();
        for p in plan {
            let key = format!(
                "{}/{}/{}/tp{}",
                p.profile.engine, p.profile.model, p.profile.gpu, p.profile.tp
            );
            *per_profile.entry(key).or_insert(0) += 1;
        }
        PlanSummary {
            name: self.name.clone(),
            configurations: self.node_groups.len(),
            nodes: self.node_groups.iter().map(|g| g.nodes).sum(),
            gpus: self
                .node_groups
                .iter()
                .map(|g| g.nodes * g.gpus_per_node)
                .sum(),
            gpus_used: plan.iter().map(|p| p.gpu_indices.len() as u32).sum(),
            instances: plan.len(),
            per_profile,
            uncalibrated_instances: plan.iter().filter(|p| p.profile.uncalibrated).count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_256_instances_on_9_configs() {
        let spec = ClusterSpec::default_256();
        let plan = spec.plan(&Catalog::builtin()).unwrap();
        let s = spec.summarize(&plan);
        assert_eq!(s.instances, 256);
        assert_eq!(s.configurations, 9);
        assert_eq!(s.gpus, 288);
        assert_eq!(s.gpus_used, 288);
        let ids: std::collections::BTreeSet<_> = plan.iter().map(|p| &p.id).collect();
        assert_eq!(ids.len(), 256);
    }

    #[test]
    fn over_subscribed_group_is_refused() {
        let spec = ClusterSpec::from_yaml(
            "name: x\nnode_groups:\n  - {name: a, gpu: H100-SXM-80GB, gpus_per_node: 8, \
             nodes: 1, deployments: [{engine: vllm, model: llama-3.1-70b, tp: 4, \
             instances: 3}]}\n",
        )
        .unwrap();
        let err = spec.plan(&Catalog::builtin()).unwrap_err();
        assert!(err.to_string().contains("not enough GPUs"), "{err}");
    }

    #[test]
    fn model_that_does_not_fit_is_refused() {
        let spec = ClusterSpec::from_yaml(
            "name: x\nnode_groups:\n  - {name: a, gpu: L4-24GB, gpus_per_node: 2, \
             nodes: 1, deployments: [{engine: vllm, model: llama-3.1-70b, tp: 2, \
             instances: 1}]}\n",
        )
        .unwrap();
        assert!(matches!(
            spec.plan(&Catalog::builtin()),
            Err(Error::DoesNotFit { .. })
        ));
    }
}
