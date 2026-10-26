//! The emulated cluster: virtual engines served over HTTP on the wall
//! clock, and PyLet sim workers that start them when a real PyLet head
//! assigns instances.
//!
//! - [`engine_server`]: one virtual engine instance (OpenAI, images,
//!   generic routes; health, version, metrics; faults).
//! - [`pylet`]: a simulated PyLet worker per node.
//! - [`host`]: all nodes of a cluster spec in one process, plus the
//!   control API.
//! - [`command`]: launch command parsing.

pub mod command;
pub mod engine_server;
pub mod host;
pub mod pylet;
