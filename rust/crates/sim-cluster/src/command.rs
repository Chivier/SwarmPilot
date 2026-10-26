//! Turn the launch command PyLet hands a worker into what the simulator
//! needs: which engine, which model, how many GPUs.
//!
//! Commands come from the engine adapters' `launch_spec().render_shell()`
//! (`vllm serve <model> ... --tensor-parallel-size N`,
//! `python -m sglang.launch_server --model-path <model> ... --tp-size N`,
//! `python -m swarmpilot.engines.servers.diffusers_server --model <model>`).
//! Anything else runs as a generic engine. The instance's environment can
//! override the result with `SWARMPILOT_SIM_ENGINE` and
//! `SWARMPILOT_SIM_MODEL`.

use std::collections::HashMap;

/// What a launch command asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub engine: String,
    /// The model id as given (catalog name or Hugging Face id).
    pub model: String,
    /// Tensor parallelism, when the command says.
    pub tp: Option<u32>,
}

pub const ENV_ENGINE: &str = "SWARMPILOT_SIM_ENGINE";
pub const ENV_MODEL: &str = "SWARMPILOT_SIM_MODEL";
const GENERIC_MODEL: &str = "generic-small";

fn flag<'a>(argv: &'a [String], names: &[&str]) -> Option<&'a str> {
    argv.iter().enumerate().find_map(|(i, a)| {
        names.iter().find_map(|n| {
            if a == n {
                argv.get(i + 1).map(String::as_str)
            } else {
                a.strip_prefix(&format!("{n}="))
            }
        })
    })
}

/// Parse a launch command, then apply environment overrides.
pub fn parse(command: &str, env: &HashMap<String, String>) -> Launch {
    let argv = shell_words::split(command)
        .unwrap_or_else(|_| command.split_whitespace().map(str::to_string).collect());
    let tp = |names: &[&str]| flag(&argv, names).and_then(|v| v.parse().ok());
    let pos = |s: &str| {
        argv.iter()
            .position(|a| a == s || a.ends_with(&format!("/{s}")))
    };

    let mut launch =
        if let Some(i) = pos("vllm").filter(|&i| argv.get(i + 1).is_some_and(|a| a == "serve")) {
            Launch {
                engine: "vllm".into(),
                model: argv
                    .get(i + 2)
                    .cloned()
                    .or_else(|| flag(&argv, &["--model"]).map(str::to_string))
                    .unwrap_or_default(),
                tp: tp(&["--tensor-parallel-size", "-tp"]),
            }
        } else if argv.iter().any(|a| a == "sglang.launch_server") {
            Launch {
                engine: "sglang".into(),
                model: flag(&argv, &["--model-path", "--model"])
                    .unwrap_or_default()
                    .to_string(),
                tp: tp(&["--tp-size", "--tp"]),
            }
        } else if argv.iter().any(|a| a.ends_with("diffusers_server")) {
            Launch {
                engine: "diffusers".into(),
                model: flag(&argv, &["--model"]).unwrap_or_default().to_string(),
                tp: None,
            }
        } else {
            Launch {
                engine: "generic".into(),
                model: GENERIC_MODEL.into(),
                tp: None,
            }
        };
    if let Some(e) = env.get(ENV_ENGINE) {
        launch.engine = e.clone();
        if e == "generic" && !env.contains_key(ENV_MODEL) {
            launch.model = GENERIC_MODEL.into();
        }
    }
    if let Some(m) = env.get(ENV_MODEL) {
        launch.model = m.clone();
    }
    launch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cmd: &str) -> Launch {
        parse(cmd, &HashMap::new())
    }

    #[test]
    fn vllm_from_adapter_shell() {
        let l = p(
            r#"vllm serve 'meta-llama/Llama-3.1-70B-Instruct' --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 4"#,
        );
        assert_eq!(l.engine, "vllm");
        assert_eq!(l.model, "meta-llama/Llama-3.1-70B-Instruct");
        assert_eq!(l.tp, Some(4));
    }

    #[test]
    fn sglang_from_adapter_shell() {
        let l = p(
            r#"python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port "$PORT" --enable-metrics --tp-size=2"#,
        );
        assert_eq!(l.engine, "sglang");
        assert_eq!(l.model, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(l.tp, Some(2));
    }

    #[test]
    fn diffusers_from_adapter_shell() {
        let l = p(
            r#"python -m swarmpilot.engines.servers.diffusers_server --model stabilityai/stable-diffusion-xl-base-1.0 --host 0.0.0.0 --port "$PORT" --default-steps 30"#,
        );
        assert_eq!(l.engine, "diffusers");
        assert_eq!(l.model, "stabilityai/stable-diffusion-xl-base-1.0");
    }

    #[test]
    fn unknown_command_is_generic_and_env_overrides() {
        assert_eq!(p("python my_server.py --port $PORT").engine, "generic");
        let env = HashMap::from([
            (ENV_ENGINE.to_string(), "vllm".to_string()),
            (ENV_MODEL.to_string(), "llama-3.1-8b".to_string()),
        ]);
        let l = parse("python tests/perf/fake_engine.py --port $PORT", &env);
        assert_eq!(
            (l.engine.as_str(), l.model.as_str()),
            ("vllm", "llama-3.1-8b")
        );
    }
}
