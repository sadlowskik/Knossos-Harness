//! One real engine call, against a real model server.
//!
//! Every other test in this crate uses `MockEngine`. That was true for 130
//! tests and the whole life of the project: the Anthropic and Ollama backends
//! had never exchanged a packet, so a green suite said nothing about whether
//! either of them worked.
//!
//! The Python side demonstrated what that hides. Its first live execute-mode
//! run found three composing defects -- an empty reply read as a completion
//! claim, a verifier that passes vacuously over an empty change set, and a
//! whole reply arriving in a response field nothing read -- none of which any
//! unit test could have caught, and all of which one live request did.
//!
//! Configured, not mocked:
//!
//! ```text
//! KNOSSOS_LIVE_OLLAMA=http://192.168.4.103:11434
//! KNOSSOS_LIVE_MODEL=gemma4:e4b          # optional
//! ```
//!
//! Unset, these skip loudly -- and fail under `KNOSSOS_LIVE_STRICT`, so CI
//! cannot quietly stop exercising them the way the cross-language suite did.

use knossos::engine::{self, ollama::OllamaEngine, Message, Request};
use knossos::themis::{Themis, EXECUTOR_ROLE};
use knossos::tools::ToolRegistry;

fn live_engine() -> Option<OllamaEngine> {
    let Ok(base) = std::env::var("KNOSSOS_LIVE_OLLAMA") else {
        let message = "KNOSSOS_LIVE_OLLAMA is not set, so no engine was \
                       contacted and this test proved nothing";
        assert!(
            std::env::var("KNOSSOS_LIVE_STRICT").is_err(),
            "{message} (KNOSSOS_LIVE_STRICT is set)"
        );
        eprintln!("\n!!! SKIPPED: {message}\n");
        return None;
    };
    let model = std::env::var("KNOSSOS_LIVE_MODEL")
        .unwrap_or_else(|_| knossos::engine::ollama::DEFAULT_MODEL.to_string());
    Some(OllamaEngine::new(model).with_base_url(base))
}

#[tokio::test]
async fn a_real_engine_answers_a_plain_question() {
    let Some(eng) = live_engine() else { return };

    let req = Request::new(
        "You are terse. Answer in one short sentence.".to_string(),
        vec![Message::user_text("What is 2 + 2?".to_string())],
    )
    .with_max_tokens(256);

    let resp = engine::complete(&eng, &req)
        .await
        .expect("live request failed");

    assert!(
        !resp.text().trim().is_empty(),
        "a live reply must contain text; got {:?}",
        resp.text()
    );
}

#[tokio::test]
async fn a_real_engine_can_be_given_the_tool_schema() {
    // The request shape the executor actually sends. A backend that serialises
    // `tools` wrongly fails here and passes every mocked test in the crate.
    let Some(eng) = live_engine() else { return };

    let themis = Themis::from_text("Be correct.");
    let req = Request::new(
        themis.system_prompt(EXECUTOR_ROLE, None),
        vec![Message::user_text(
            "Read the file src/lib.rs. Use a tool.".to_string(),
        )],
    )
    .with_tools(ToolRegistry::standard().defs())
    .with_max_tokens(512);

    let resp = engine::complete(&eng, &req)
        .await
        .expect("live request failed");

    // Whether the model *chooses* to call a tool is its business; that the
    // round trip survives a tools payload is ours.
    let produced_something = !resp.text().trim().is_empty() || !resp.tool_uses().is_empty();
    assert!(
        produced_something,
        "the model returned neither text nor a tool call"
    );
}
