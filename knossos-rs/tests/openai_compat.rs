//! Deterministic wire-level checks for OpenRouter and other OpenAI-compatible
//! providers. No model or internet connection is used.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use knossos::engine::openai::OpenAICompatEngine;
use knossos::engine::{self, EngineError, Message, Request, StopReason, StreamDelta};
use knossos::resilience::{Policy, Resilient};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Clone)]
struct Reply {
    status: u16,
    content_type: &'static str,
    body: &'static str,
    delay: Duration,
}

impl Reply {
    fn json(status: u16, body: &'static str) -> Self {
        Self {
            status,
            content_type: "application/json",
            body,
            delay: Duration::ZERO,
        }
    }

    fn sse(body: &'static str) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body,
            delay: Duration::ZERO,
        }
    }

    fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

async fn mock_provider(
    replies: Vec<Reply>,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            captured.lock().unwrap().push(request);
            if !reply.delay.is_zero() {
                tokio::time::sleep(reply.delay).await;
            }
            let reason = match reply.status {
                200 => "OK",
                400 => "Bad Request",
                401 => "Unauthorized",
                429 => "Too Many Requests",
                _ => "Error",
            };
            let head = format!(
                "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                reply.status,
                reason,
                reply.content_type,
                reply.body.len()
            );
            // A deadline test intentionally disconnects before this write.
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(reply.body.as_bytes()).await;
        }
    });
    (format!("http://{address}/v1"), requests, handle)
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut target = None;
    loop {
        let count = socket.read(&mut chunk).await.unwrap();
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if target.is_none() {
            if let Some(head_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&bytes[..head_end]);
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                target = Some(head_end + 4 + length);
            }
        }
        if target.is_some_and(|needed| bytes.len() >= needed) {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn request() -> Request {
    Request::new(
        "Be concise.",
        vec![Message::user_text("Say hello or call a tool.")],
    )
    .with_max_tokens(512)
}

const TEXT_REPLY: &str = r#"{
  "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 7, "completion_tokens": 1}
}"#;

#[tokio::test]
async fn a_non_stream_round_trip_carries_auth_and_usage() {
    let (base, requests, server) = mock_provider(vec![Reply::json(200, TEXT_REPLY)]).await;
    let provider = OpenAICompatEngine::new(
        "openrouter",
        "test/model",
        base,
        Some("test-bearer-value".into()),
        1024,
    );

    let response = engine::complete(&provider, &request()).await.unwrap();
    server.await.unwrap();

    assert_eq!(response.text(), "hello");
    assert_eq!(response.usage.input_tokens, 7);
    assert_eq!(response.usage.output_tokens, 1);
    let request = &requests.lock().unwrap()[0];
    assert!(request.contains("authorization: Bearer test-bearer-value"));
    assert!(request.contains("POST /v1/chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn streamed_text_and_tool_arguments_are_reassembled() {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"looking\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"src/lib.rs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (base, _, server) = mock_provider(vec![Reply::sse(body)]).await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);
    let deltas = Mutex::new(Vec::new());

    let response = engine::complete_stream(&provider, &request(), &|delta| {
        deltas.lock().unwrap().push(delta)
    })
    .await
    .unwrap();
    server.await.unwrap();

    assert_eq!(
        deltas.into_inner().unwrap(),
        vec![StreamDelta::Text("looking".into())]
    );
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let calls = response.tool_uses();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, "read_file");
    assert_eq!(calls[0].2["path"], "src/lib.rs");
}

#[tokio::test]
async fn a_stream_shape_refusal_falls_back_once_and_sticks() {
    let (base, requests, server) = mock_provider(vec![
        Reply::json(400, r#"{"error":{"message":"stream is unsupported"}}"#),
        Reply::json(200, TEXT_REPLY),
        Reply::json(200, TEXT_REPLY),
    ])
    .await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);

    let first = engine::complete_stream(&provider, &request(), &|_| {})
        .await
        .unwrap();
    let second = engine::complete_stream(&provider, &request(), &|_| {})
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(first.text(), "hello");
    assert_eq!(second.text(), "hello");
    let requests = requests.lock().unwrap();
    assert!(requests[0].contains("\"stream\":true"));
    assert!(!requests[1].contains("\"stream\":true"));
    assert!(!requests[2].contains("\"stream\":true"));
}

#[tokio::test]
async fn a_stream_rate_limit_is_not_doubled_by_fallback() {
    let (base, requests, server) = mock_provider(vec![Reply::json(
        429,
        r#"{"error":{"message":"rate limited"}}"#,
    )])
    .await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);

    let error = engine::complete_stream(&provider, &request(), &|_| {})
        .await
        .unwrap_err();
    server.await.unwrap();

    assert!(matches!(
        error.downcast_ref::<EngineError>(),
        Some(EngineError::Status { status: 429, .. })
    ));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn resilience_retries_one_rate_limit_then_succeeds() {
    let (base, requests, server) = mock_provider(vec![
        Reply::json(429, r#"{"error":{"message":"rate limited"}}"#),
        Reply::json(200, TEXT_REPLY),
    ])
    .await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);
    let resilient = Resilient::with_policy(
        Box::new(provider),
        Policy {
            max_attempts: 2,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            trip_after: 3,
            cooldown: Duration::from_secs(1),
        },
    );

    let response = engine::complete(&resilient, &request()).await.unwrap();
    server.await.unwrap();

    assert_eq!(response.text(), "hello");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn an_unresponsive_provider_hits_the_total_deadline() {
    let (base, _, server) = mock_provider(vec![
        Reply::json(200, TEXT_REPLY).delayed(Duration::from_millis(200))
    ])
    .await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024)
        .with_timeout(Duration::from_millis(25));

    let error = engine::complete(&provider, &request()).await.unwrap_err();
    server.await.unwrap();

    assert!(matches!(
        error.downcast_ref::<EngineError>(),
        Some(EngineError::Transport { .. })
    ));
}

#[tokio::test]
async fn malformed_success_is_reported_without_an_unchanged_retry() {
    let (base, requests, server) = mock_provider(vec![Reply::json(200, r#"{"choices":["#)]).await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);

    let error = engine::complete(&provider, &request()).await.unwrap_err();
    server.await.unwrap();

    assert!(error
        .to_string()
        .contains("decoding OpenAI-compat response"));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_context_limit_shrinks_output_before_the_retry() {
    let body = r#"{"error":{"message":"token limit 1600, requested 1800"}}"#;
    let (base, requests, server) =
        mock_provider(vec![Reply::json(413, body), Reply::json(200, TEXT_REPLY)]).await;
    let provider = OpenAICompatEngine::new("openrouter", "test/model", base, None, 1024);
    let limited_request = request().with_max_tokens(1024);

    let response = engine::complete(&provider, &limited_request).await.unwrap();
    server.await.unwrap();

    assert_eq!(response.text(), "hello");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let first: serde_json::Value =
        serde_json::from_str(requests[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let second: serde_json::Value =
        serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(first["max_tokens"], 1024);
    assert!(second["max_tokens"].as_u64().unwrap() < 1024);
    assert!(second["max_tokens"].as_u64().unwrap() >= 512);
}
