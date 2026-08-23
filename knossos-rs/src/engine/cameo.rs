//! Cameo node as an engine slot.
//!
//! OpenAI-compat `/v1` is the inference door. This wrapper is the operator
//! seam around it: discover what is resident (`GET /api/engines`, then
//! `GET /v1/models`), and — only with the host-only unix socket or a console
//! key — `POST /api/servers` to bring a cold model up. A consumer key never
//! loads VRAM. Fail closed if the model is not running and we cannot ensure.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::engine::openai::OpenAICompatEngine;
use crate::engine::types::{Request, Response};
use crate::engine::Engine;

const PROVIDER: &str = "Cameo";
const DEFAULT_ORIGIN: &str = "http://127.0.0.1:9090";
const DEFAULT_SOCKET: &str = "/run/cameo/cameo.sock";
const ENSURE_POLLS: usize = 120;
const ENSURE_WAIT: Duration = Duration::from_millis(500);
const RESIDENCY_CACHE: Duration = Duration::from_secs(2);

/// Cameo-backed engine: ensure the model is resident, then speak `/v1`.
pub struct CameoEngine {
    inner: OpenAICompatEngine,
    origin: String,
    model: String,
    serve_key: Option<String>,
    console_key: Option<String>,
    socket: Option<PathBuf>,
    ensure_port: u16,
    client: reqwest::Client,
    ensured_at: Mutex<Option<Instant>>,
}

impl CameoEngine {
    /// `base_url` is the OpenAI-compat root (`http://host:9090/v1`).
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        serve_key: Option<String>,
        max_tokens: u32,
    ) -> Self {
        let model = model.into();
        let base_url = base_url.into();
        let origin = origin_from_v1(&base_url);
        let console_key = std::env::var("CAMEO_CONSOLE_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let socket = std::env::var("CAMEO_SOCKET")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let ensure_port = std::env::var("CAMEO_ENSURE_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8080);
        CameoEngine {
            inner: OpenAICompatEngine::new(
                "cameo",
                model.clone(),
                base_url,
                serve_key.clone(),
                max_tokens,
            ),
            origin,
            model,
            serve_key,
            console_key,
            socket,
            ensure_port,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(2))
                .timeout(Duration::from_secs(10))
                .build()
                .expect("valid Cameo HTTP client"),
            ensured_at: Mutex::new(None),
        }
    }

    /// Override the operator seam (tests, or a caller that already resolved keys).
    pub fn with_operator(mut self, console_key: Option<String>, socket: Option<PathBuf>) -> Self {
        self.console_key = console_key;
        self.socket = socket;
        self
    }

    /// Cheap GET. Does not try to start a cold model.
    pub async fn is_resident(&self) -> bool {
        listed(&self.listed_models().await.unwrap_or_default(), &self.model)
    }

    /// Discover, optionally ensure, fail closed if the model is still cold.
    pub async fn ensure(&self) -> Result<()> {
        let mut g = self.ensured_at.lock().await;
        if g.is_some_and(|at| at.elapsed() < RESIDENCY_CACHE) {
            return Ok(());
        }
        if listed(&self.listed_models().await?, &self.model) {
            *g = Some(Instant::now());
            return Ok(());
        }
        self.try_start().await?;
        if listed(&self.listed_models().await?, &self.model) {
            *g = Some(Instant::now());
            return Ok(());
        }
        for _ in 0..ENSURE_POLLS {
            tokio::time::sleep(ENSURE_WAIT).await;
            if listed(&self.listed_models().await?, &self.model) {
                *g = Some(Instant::now());
                return Ok(());
            }
        }
        bail!(fail_closed(&self.model));
    }

    async fn listed_models(&self) -> Result<Vec<String>> {
        if let Some(models) = self.engines_models().await? {
            return Ok(models);
        }
        self.v1_models().await
    }

    async fn engines_models(&self) -> Result<Option<Vec<String>>> {
        if let Some((status, body)) = self.operator_get("/api/engines").await? {
            if (200..300).contains(&status) {
                validate_engine_contract(&body)?;
                if let Some(window) = context_from_engines(&body, &self.model) {
                    self.inner.bound_context_window(window);
                }
                return Ok(Some(models_from_engines(&body)));
            }
        }
        // Consumer key (or open inference) can still read the descriptor.
        let url = format!("{}/api/engines", self.origin);
        let discovery_key = self.serve_key.as_deref().or(self.console_key.as_deref());
        let (status, body) = http_get(&self.client, &url, discovery_key).await?;
        if (200..300).contains(&status) {
            validate_engine_contract(&body)?;
            if let Some(window) = context_from_engines(&body, &self.model) {
                self.inner.bound_context_window(window);
            }
            return Ok(Some(models_from_engines(&body)));
        }
        Ok(None)
    }

    async fn v1_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.origin);
        let discovery_key = self.serve_key.as_deref().or(self.console_key.as_deref());
        let (status, body) = http_get(&self.client, &url, discovery_key).await?;
        if !(200..300).contains(&status) {
            bail!("{PROVIDER} GET /v1/models returned {status}: {body}");
        }
        Ok(models_from_v1(&body))
    }

    async fn try_start(&self) -> Result<()> {
        let payload = json!({
            "model": self.model,
            "host": "127.0.0.1",
            "port": self.ensure_port,
        });
        let body = payload.to_string();

        if let Some((status, text)) = self.operator_post("/api/servers", body.as_bytes()).await? {
            return interpret_start(status, &text, &self.model);
        }
        if let Some(key) = &self.console_key {
            let url = format!("{}/api/servers", self.origin);
            let (status, text) = http_post(&self.client, &url, Some(key), &payload).await?;
            return interpret_start(status, &text, &self.model);
        }
        bail!(fail_closed(&self.model));
    }

    async fn operator_get(&self, path: &str) -> Result<Option<(u16, String)>> {
        self.unix_http("GET", path, None).await
    }

    async fn operator_post(&self, path: &str, body: &[u8]) -> Result<Option<(u16, String)>> {
        self.unix_http("POST", path, Some(body)).await
    }

    async fn unix_http(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<Option<(u16, String)>> {
        #[cfg(unix)]
        {
            let socket = match self.socket_path() {
                Some(p) => p,
                None => return Ok(None),
            };
            return unix_request(&socket, method, path, body)
                .await
                .map(Some)
                .map_err(|e| anyhow::anyhow!("{PROVIDER} unix {method} {path}: {e}"));
        }
        #[cfg(not(unix))]
        {
            let _ = (method, path, body, &self.socket);
            Ok(None)
        }
    }

    #[cfg(unix)]
    fn socket_path(&self) -> Option<PathBuf> {
        if let Some(p) = &self.socket {
            return Some(p.clone());
        }
        let p = PathBuf::from(DEFAULT_SOCKET);
        p.exists().then_some(p)
    }
}

#[async_trait]
impl Engine for CameoEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        self.ensure().await?;
        self.inner.complete(req).await
    }

    async fn complete_stream(
        &self,
        req: &Request,
        on_delta: &(dyn Fn(crate::engine::StreamDelta) + Send + Sync),
    ) -> Result<Response> {
        self.ensure().await?;
        self.inner.complete_stream(req, on_delta).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn supports_native_tools(&self) -> bool {
        self.inner.supports_native_tools()
    }

    fn restore_limits(&self) -> bool {
        self.inner.restore_limits()
    }

    fn context_window(&self) -> Option<u32> {
        self.inner.context_window()
    }
}

/// Strip a trailing `/v1` so control-plane paths hang off the daemon origin.
pub fn origin_from_v1(base: &str) -> String {
    let t = base.trim().trim_end_matches('/');
    let stripped = t.strip_suffix("/v1").unwrap_or(t).trim_end_matches('/');
    if stripped.is_empty() {
        DEFAULT_ORIGIN.to_string()
    } else {
        stripped.to_string()
    }
}

fn listed(models: &[String], want: &str) -> bool {
    let want = stem(want);
    models.iter().any(|m| stem(m) == want)
}

fn stem(s: &str) -> &str {
    s.trim().trim_end_matches(".gguf")
}

fn models_from_engines(body: &str) -> Vec<String> {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("models").cloned())
        .and_then(|m| m.as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Accept the pre-versioned legacy descriptor, but fail closed on a version or
/// route the client does not understand. Silently treating a future contract as
/// v1 could provision or route the wrong endpoint.
fn validate_engine_contract(body: &str) -> Result<()> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("{PROVIDER} engine descriptor is not JSON: {e}"))?;
    if let Some(version) = value.get("contract_version").and_then(Value::as_str) {
        if version != "cameo-engine/v1" {
            bail!(
                "{PROVIDER} engine contract '{version}' is unsupported; expected cameo-engine/v1"
            );
        }
    }
    if let Some(path) = value.get("openai_base_path").and_then(Value::as_str) {
        if path != "/v1" {
            bail!("{PROVIDER} engine descriptor advertises unsupported base path '{path}'");
        }
    }
    if !value.get("models").is_some_and(Value::is_array) {
        bail!("{PROVIDER} engine descriptor is missing the models array");
    }
    Ok(())
}

fn models_from_v1(body: &str) -> Vec<String> {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("data").cloned())
        .and_then(|m| m.as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.get("id").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn context_from_engines(body: &str, model: &str) -> Option<u32> {
    let value: Value = serde_json::from_str(body).ok()?;
    let want = stem(model);
    value
        .get("model_profiles")?
        .as_array()?
        .iter()
        .find(|profile| {
            profile
                .get("model")
                .and_then(Value::as_str)
                .is_some_and(|name| stem(name) == want)
        })?
        .get("context_tokens")?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
}

fn interpret_start(status: u16, text: &str, model: &str) -> Result<()> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    if status == 409 {
        let error = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        let same_endpoint = error.to_ascii_lowercase().contains("already running")
            && error
                .to_ascii_lowercase()
                .contains(&stem(model).to_ascii_lowercase());
        if same_endpoint {
            return Ok(());
        }
    }
    bail!("{PROVIDER} POST /api/servers for '{model}' returned {status}: {text}");
}

fn fail_closed(model: &str) -> String {
    format!(
        "{PROVIDER} has no running endpoint for '{model}'. Start it from the console, \
         run Knossos on the same box (unix socket {DEFAULT_SOCKET}), or set \
         CAMEO_CONSOLE_KEY so the harness can POST /api/servers."
    )
}

async fn http_get(client: &reqwest::Client, url: &str, key: Option<&str>) -> Result<(u16, String)> {
    let mut req = client.get(url);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{PROVIDER}: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    Ok((status, text))
}

async fn http_post(
    client: &reqwest::Client,
    url: &str,
    key: Option<&str>,
    body: &Value,
) -> Result<(u16, String)> {
    let mut req = client.post(url).json(body);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{PROVIDER}: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    Ok((status, text))
}

#[cfg(unix)]
async fn unix_request(
    socket: &PathBuf,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<(u16, String)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    let mut stream = UnixStream::connect(socket)
        .await
        .map_err(|e| anyhow::anyhow!("connecting to {}: {e}", socket.display()))?;
    let payload = body.unwrap_or(b"");
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: cameo\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).await?;
    if !payload.is_empty() {
        stream.write_all(payload).await?;
    }
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await?;
    parse_http(&buf)
}

#[cfg_attr(not(unix), allow(dead_code))]
fn parse_http(bytes: &[u8]) -> Result<(u16, String)> {
    let text = String::from_utf8_lossy(bytes);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .or_else(|| text.split_once("\n\n"))
        .ok_or_else(|| anyhow::anyhow!("{PROVIDER} reply had no header break"))?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("{PROVIDER} reply had no status"))?;
    Ok((status, body.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn origin_strips_v1_and_trailing_slash() {
        assert_eq!(
            origin_from_v1("http://127.0.0.1:9090/v1"),
            "http://127.0.0.1:9090"
        );
        assert_eq!(origin_from_v1("http://box:9090/v1/"), "http://box:9090");
        assert_eq!(origin_from_v1("http://box:9090"), "http://box:9090");
    }

    #[test]
    fn listed_matches_gguf_stem() {
        let models = vec!["qwen2.5-0.5b.gguf".into()];
        assert!(listed(&models, "qwen2.5-0.5b"));
        assert!(listed(&models, "qwen2.5-0.5b.gguf"));
        assert!(!listed(&models, "tinyllama"));
    }

    #[test]
    fn engines_body_parses_models() {
        let body = r#"{"node":"box","openai_base_path":"/v1","models":["qwen2.5-0.5b"]}"#;
        assert_eq!(models_from_engines(body), vec!["qwen2.5-0.5b"]);
        assert!(models_from_engines("{}").is_empty());
    }

    #[test]
    fn versioned_contract_is_validated_and_legacy_is_still_readable() {
        let v1 = r#"{
            "contract_version":"cameo-engine/v1",
            "openai_base_path":"/v1",
            "models":[]
        }"#;
        assert!(validate_engine_contract(v1).is_ok());
        assert!(validate_engine_contract(r#"{"models":[]}"#).is_ok());
        assert!(
            validate_engine_contract(r#"{"contract_version":"cameo-engine/v2","models":[]}"#)
                .is_err()
        );
        assert!(validate_engine_contract(
            r#"{"contract_version":"cameo-engine/v1","openai_base_path":"/other","models":[]}"#
        )
        .is_err());
    }

    #[test]
    fn parse_http_reads_status_and_body() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        let (s, b) = parse_http(raw).unwrap();
        assert_eq!(s, 200);
        assert_eq!(b, "{}");
    }

    async fn spawn_http<F>(handler: F) -> String
    where
        F: Fn(&str, &str, &str) -> (u16, String) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let first = req.lines().next().unwrap_or("");
                let mut parts = first.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("/").to_string();
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                let (status, resp_body) = handler(&method, &path, &body);
                let reason = if (200..300).contains(&status) {
                    "OK"
                } else {
                    "ERR"
                };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn ensure_reuses_a_resident_model() {
        let origin = spawn_http(|method, path, _| {
            assert_eq!(method, "GET");
            assert_eq!(path, "/api/engines");
            (200, r#"{"models":["qwen2.5-0.5b"]}"#.into())
        })
        .await;
        let engine = CameoEngine::new("qwen2.5-0.5b", format!("{origin}/v1"), None, 512)
            .with_operator(None, None);
        engine.ensure().await.unwrap();
    }

    #[tokio::test]
    async fn ensure_fails_closed_when_cold_and_no_operator() {
        let origin = spawn_http(|_m, path, _| {
            if path == "/api/engines" {
                (200, r#"{"models":[]}"#.into())
            } else {
                (404, "{}".into())
            }
        })
        .await;
        let engine = CameoEngine::new("qwen2.5-0.5b", format!("{origin}/v1"), None, 512)
            .with_operator(None, None);
        let err = engine.ensure().await.unwrap_err().to_string();
        assert!(err.contains("no running endpoint"), "{err}");
        assert!(err.contains("CAMEO_CONSOLE_KEY"), "{err}");
    }

    #[tokio::test]
    async fn ensure_posts_servers_with_console_key() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let posted = Arc::new(AtomicBool::new(false));
        let flag = posted.clone();
        let origin = spawn_http(move |method, path, _| {
            if method == "POST" && path == "/api/servers" {
                flag.store(true, Ordering::SeqCst);
                (201, r#"{"id":"qwen-8080","state":"running"}"#.into())
            } else if path == "/api/engines" {
                if flag.load(Ordering::SeqCst) {
                    (200, r#"{"models":["qwen2.5-0.5b"]}"#.into())
                } else {
                    (200, r#"{"models":[]}"#.into())
                }
            } else {
                (404, "{}".into())
            }
        })
        .await;
        let engine = CameoEngine::new("qwen2.5-0.5b", format!("{origin}/v1"), None, 512)
            .with_operator(Some("console-key".into()), None);
        engine.ensure().await.unwrap();
        assert!(posted.load(Ordering::SeqCst), "expected POST /api/servers");
    }
}
