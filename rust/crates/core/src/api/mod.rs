//! API style translation (`swarmpilot/api`): decode a client's request in
//! one API style, re-encode it for an engine in another, and translate the
//! answer and its SSE stream back. Python's dynamic semantics (truthiness,
//! `dict.get`, `str()`, `json.dumps`) are emulated in [`py`] so that the
//! same input gives the same bytes; `tests/core/test_translation_parity.py`
//! compares the two.

pub mod anthropic_messages;
pub mod base;
pub mod canonical;
pub mod openai_chat;
pub mod openai_responses;
pub mod py;
pub mod sse;
pub mod style;
pub mod translate;
