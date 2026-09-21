//! Ephemeral authority for one Field server process. Port of
//! `field/server/src/security.js`.
//!
//! Field listens on loopback only, and the browser earns its cookie exactly
//! once, from a bootstrap link printed at start. Browser and harness
//! credentials are separate on purpose: a harness token grants the one
//! internal permission endpoint for one session and nothing else, and the
//! operator's cookie grants nothing to a harness. State-changing requests
//! also need the exact local `Origin`, which is the whole CSRF defence.
//!
//! The functions here decide; they never touch a socket. The server maps
//! the decisions to responses, so the policy is testable request by request
//! without one.

use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;
use url::Url;

const TOKEN_BYTES: usize = 32;
pub const COOKIE_NAME: &str = "field_session";
const UNSAFE_METHODS: [&str; 4] = ["POST", "PUT", "PATCH", "DELETE"];
const LOOPBACK_HOSTS: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

fn base64url(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// 32 random bytes, base64url: what both tokens and every harness capability are.
pub fn generate_token() -> String {
    let mut bytes = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut bytes).expect("the OS random source is available");
    base64url(&bytes)
}

/// Constant-time equality for secrets of the same length.
fn equal_secret(left: &str, right: &str) -> bool {
    let (a, b) = (left.as_bytes(), right.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn cookie_value<'a>(header: Option<&'a str>, name: &str) -> Option<&'a str> {
    let header = header?;
    for part in header.split(';') {
        let Some(index) = part.find('=') else {
            continue;
        };
        if index < 1 {
            continue;
        }
        if part[..index].trim() == name {
            return Some(part[index + 1..].trim());
        }
    }
    None
}

pub fn is_loopback_address(address: &str) -> bool {
    let normalized = address.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "127.0.0.1" | "::1" | "::ffff:127.0.0.1"
    )
}

/// `host:port` when the Host header names loopback on exactly `port`.
fn accepted_authority(host: Option<&str>, port: u16) -> Option<String> {
    let host = host?;
    if host
        .chars()
        .any(|c| c.is_whitespace() || matches!(c, '/' | '@' | '\\'))
    {
        return None;
    }
    let parsed = Url::parse(&format!("http://{host}")).ok()?;
    let hostname = parsed.host_str()?.to_ascii_lowercase();
    if !LOOPBACK_HOSTS.contains(&hostname.as_str()) {
        return None;
    }
    // A Host without an explicit port never matches: the reference compares
    // the port *string*, and an implicit port is the empty string.
    if parsed.port() != Some(port) {
        return None;
    }
    Some(format!("{hostname}:{port}"))
}

fn structurally_plain_http(parsed: &Url) -> bool {
    parsed.scheme() == "http"
        && parsed.username().is_empty()
        && parsed.password().is_none()
        && parsed.path() == "/"
        && parsed.query().is_none()
        && parsed.fragment().is_none()
}

/// `http://host:port` for an exact loopback HTTP origin, else `None`.
fn normalized_loopback_origin(value: &str) -> Option<String> {
    let parsed = Url::parse(value).ok()?;
    if !structurally_plain_http(&parsed) {
        return None;
    }
    let hostname = parsed.host_str()?.to_ascii_lowercase();
    if !LOOPBACK_HOSTS.contains(&hostname.as_str()) {
        return None;
    }
    let port = parsed.port()?;
    Some(format!("http://{hostname}:{port}"))
}

fn origin_of(parsed: &Url) -> Option<String> {
    let host = parsed.host_str()?.to_ascii_lowercase();
    let port = parsed.port()?;
    Some(format!("{}://{host}:{port}", parsed.scheme()))
}

fn same_origin(req: &RequestFacts, port: u16, trusted: &[String]) -> bool {
    let Some(authority) = accepted_authority(req.host.as_deref(), port) else {
        return false;
    };
    let Some(origin) = req.origin.as_deref() else {
        return false;
    };
    let Ok(parsed) = Url::parse(origin) else {
        return false;
    };
    if !structurally_plain_http(&parsed) {
        return false;
    }
    let host_port = match (parsed.host_str(), parsed.port()) {
        (Some(h), Some(p)) => format!("{}:{p}", h.to_ascii_lowercase()),
        _ => return false,
    };
    host_port == authority || origin_of(&parsed).is_some_and(|o| trusted.contains(&o))
}

fn bearer_token(header: Option<&str>) -> Option<&str> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^Bearer ([A-Za-z0-9_-]+)$").expect("static regex"));
    re.captures(header?)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str())
}

fn capability_key(token: &str) -> String {
    base64url(&Sha256::digest(token.as_bytes()))
}

/// What the server needs to know about a request to decide. Header values
/// are passed as received; `remote_address` is the peer IP as text.
#[derive(Debug, Clone, Default)]
pub struct RequestFacts {
    pub method: String,
    pub path: String,
    /// The `bootstrap` query parameter, `Some` when present (even if empty).
    pub bootstrap: Option<String>,
    pub host: Option<String>,
    pub origin: Option<String>,
    pub cookie: Option<String>,
    pub authorization: Option<String>,
    pub content_type: Option<String>,
    pub remote_address: String,
}

/// A refusal, with the status and stable code the HTTP layer sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denied {
    pub status: u16,
    pub code: &'static str,
    pub error: &'static str,
}

fn denied(status: u16, code: &'static str, error: &'static str) -> Denied {
    Denied {
        status,
        code,
        error,
    }
}

/// Who a request is acting as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Authority {
    Operator,
    Harness { session_id: String },
}

/// Outcome of the bootstrap handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bootstrap {
    /// Not a bootstrap request; carry on.
    NotHandled,
    /// Redirect there, setting the cookie if given.
    Granted {
        redirect: String,
        cookie: Option<String>,
    },
    Denied(Denied),
}

#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Field security requires a valid listening port")]
    InvalidPort,
    #[error("Field trusted origins must be exact loopback HTTP origins")]
    TrustedOrigin,
    #[error("Field bootstrap redirect must be / or an exact loopback HTTP origin")]
    BootstrapRedirect,
    #[error("Field frame sources must be exact HTTP(S) origins")]
    FrameSource,
}

pub type Clock = Box<dyn Fn() -> i64 + Send + Sync>;

pub struct SecurityOptions {
    pub port: u16,
    pub bootstrap_token: Option<String>,
    pub browser_token: Option<String>,
    pub trusted_origins: Vec<String>,
    pub bootstrap_redirect: String,
    pub frame_sources: Vec<String>,
    pub bootstrap_ttl_ms: i64,
    pub now: Option<Clock>,
}

impl SecurityOptions {
    pub fn new(port: u16) -> Self {
        SecurityOptions {
            port,
            bootstrap_token: None,
            browser_token: None,
            trusted_origins: Vec::new(),
            bootstrap_redirect: "/".into(),
            frame_sources: Vec::new(),
            bootstrap_ttl_ms: 10 * 60 * 1000,
            now: None,
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub struct ControlSecurity {
    port: u16,
    bootstrap_token: String,
    browser_token: String,
    trusted_origins: Vec<String>,
    redirect: String,
    frame_sources: Vec<String>,
    bootstrap_consumed: bool,
    browser_enabled: bool,
    bootstrap_expires_at: i64,
    harness: HashMap<String, String>,
    now: Clock,
}

impl std::fmt::Debug for ControlSecurity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlSecurity")
            .field("port", &self.port)
            .field("bootstrap_consumed", &self.bootstrap_consumed)
            .field("browser_enabled", &self.browser_enabled)
            .field("harness_tokens", &self.harness.len())
            .finish()
    }
}

impl ControlSecurity {
    pub fn new(options: SecurityOptions) -> Result<Self, SecurityError> {
        if options.port == 0 {
            return Err(SecurityError::InvalidPort);
        }
        let mut trusted_origins = Vec::new();
        for origin in &options.trusted_origins {
            trusted_origins
                .push(normalized_loopback_origin(origin).ok_or(SecurityError::TrustedOrigin)?);
        }
        let redirect = if options.bootstrap_redirect == "/" {
            "/".to_string()
        } else {
            normalized_loopback_origin(&options.bootstrap_redirect)
                .ok_or(SecurityError::BootstrapRedirect)?
        };
        let mut frame_sources = Vec::new();
        for value in &options.frame_sources {
            let parsed = Url::parse(value).map_err(|_| SecurityError::FrameSource)?;
            let plain = matches!(parsed.scheme(), "http" | "https")
                && parsed.username().is_empty()
                && parsed.password().is_none()
                && parsed.path() == "/"
                && parsed.query().is_none()
                && parsed.fragment().is_none();
            if !plain {
                return Err(SecurityError::FrameSource);
            }
            frame_sources.push(parsed.origin().ascii_serialization());
        }
        let now: Clock = options.now.unwrap_or_else(|| Box::new(now_ms));
        let bootstrap_expires_at = now() + options.bootstrap_ttl_ms;
        Ok(ControlSecurity {
            port: options.port,
            bootstrap_token: options.bootstrap_token.unwrap_or_else(generate_token),
            browser_token: options.browser_token.unwrap_or_else(generate_token),
            trusted_origins,
            redirect,
            frame_sources,
            bootstrap_consumed: false,
            browser_enabled: false,
            bootstrap_expires_at,
            harness: HashMap::new(),
            now,
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn bootstrap_token(&self) -> &str {
        &self.bootstrap_token
    }

    pub fn browser_token(&self) -> &str {
        &self.browser_token
    }

    /// The one-time link printed at start.
    pub fn bootstrap_url(&self) -> String {
        format!(
            "http://127.0.0.1:{}/?bootstrap={}",
            self.port, self.bootstrap_token
        )
    }

    /// Mint the internal capability for one session, replacing any earlier one.
    pub fn mint_harness_token(&mut self, session_id: &str) -> String {
        self.revoke_harness_token(session_id);
        let token = generate_token();
        self.harness
            .insert(capability_key(&token), session_id.to_string());
        token
    }

    pub fn revoke_harness_token(&mut self, session_id: &str) {
        self.harness.retain(|_, owner| owner != session_id);
    }

    pub fn revoke_all_harness_tokens(&mut self) {
        self.harness.clear();
    }

    /// Loopback peer and an exact loopback Host, or a refusal.
    pub fn authorize_network(&self, req: &RequestFacts) -> Result<(), Denied> {
        if !is_loopback_address(&req.remote_address) {
            return Err(denied(
                403,
                "loopback_required",
                "Field only accepts loopback clients.",
            ));
        }
        if accepted_authority(req.host.as_deref(), self.port).is_none() {
            return Err(denied(
                421,
                "invalid_host",
                "Field rejected the Host header.",
            ));
        }
        Ok(())
    }

    fn has_browser_session(&self, req: &RequestFacts) -> bool {
        self.browser_enabled
            && cookie_value(req.cookie.as_deref(), COOKIE_NAME)
                .is_some_and(|value| equal_secret(value, &self.browser_token))
    }

    /// Log the browser out; returns the `Set-Cookie` value that clears it.
    pub fn revoke_browser_session(&mut self) -> String {
        self.browser_enabled = false;
        self.bootstrap_consumed = true;
        format!("{COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
    }

    /// Handle `GET /?bootstrap=...`: single use, time-limited, loopback only.
    pub fn consume_bootstrap(&mut self, req: &RequestFacts) -> Bootstrap {
        if req.path != "/" || req.bootstrap.is_none() {
            return Bootstrap::NotHandled;
        }
        if let Err(refused) = self.authorize_network(req) {
            return Bootstrap::Denied(refused);
        }
        if req.method != "GET" {
            return Bootstrap::Denied(denied(405, "method_not_allowed", "Bootstrap requires GET."));
        }
        if self.bootstrap_consumed {
            if self.has_browser_session(req) {
                return Bootstrap::Granted {
                    redirect: self.redirect.clone(),
                    cookie: None,
                };
            }
            return Bootstrap::Denied(denied(
                410,
                "bootstrap_consumed",
                "This bootstrap link has already been used. Restart Field to mint a new link.",
            ));
        }
        let presented = req.bootstrap.as_deref().unwrap_or("");
        if (self.now)() >= self.bootstrap_expires_at
            || !equal_secret(presented, &self.bootstrap_token)
        {
            return Bootstrap::Denied(denied(
                401,
                "invalid_bootstrap",
                "The bootstrap token is invalid.",
            ));
        }
        self.bootstrap_consumed = true;
        self.browser_enabled = true;
        Bootstrap::Granted {
            redirect: self.redirect.clone(),
            cookie: Some(format!(
                "{COOKIE_NAME}={}; HttpOnly; SameSite=Strict; Path=/",
                self.browser_token
            )),
        }
    }

    /// Every request after bootstrap.
    pub fn authorize_request(&self, req: &RequestFacts) -> Result<Authority, Denied> {
        self.authorize_network(req)?;

        if req.path == "/api/internal/permission" {
            if req.method != "POST" {
                return Err(denied(
                    405,
                    "method_not_allowed",
                    "The internal permission endpoint requires POST.",
                ));
            }
            let session_id = bearer_token(req.authorization.as_deref())
                .and_then(|token| self.harness.get(&capability_key(token)))
                .cloned();
            let Some(session_id) = session_id else {
                return Err(denied(
                    401,
                    "internal_auth_required",
                    "Valid harness authorization is required.",
                ));
            };
            let json = req
                .content_type
                .as_deref()
                .unwrap_or("")
                .to_ascii_lowercase()
                .starts_with("application/json");
            if !json {
                return Err(denied(
                    415,
                    "json_required",
                    "The internal permission endpoint requires JSON.",
                ));
            }
            return Ok(Authority::Harness { session_id });
        }

        if !self.has_browser_session(req) {
            return Err(denied(
                401,
                "browser_auth_required",
                "Open the fresh bootstrap URL printed by the Field server.",
            ));
        }
        if UNSAFE_METHODS.contains(&req.method.as_str())
            && !same_origin(req, self.port, &self.trusted_origins)
        {
            return Err(denied(
                403,
                "origin_required",
                "State-changing Field requests require the exact local origin.",
            ));
        }
        Ok(Authority::Operator)
    }

    /// The WebSocket handshake: an authorized GET on `/ws` from the exact origin.
    pub fn authorize_upgrade(&self, req: &RequestFacts) -> Result<Authority, Denied> {
        let authority = self.authorize_request(req)?;
        if req.path != "/ws" {
            return Err(denied(404, "not_found", "Unknown WebSocket endpoint."));
        }
        if req.method != "GET" || !same_origin(req, self.port, &self.trusted_origins) {
            return Err(denied(
                403,
                "origin_required",
                "Field WebSockets require the exact local origin.",
            ));
        }
        Ok(authority)
    }

    /// The response headers every reply carries.
    pub fn headers(&self) -> Vec<(&'static str, String)> {
        let port = self.port;
        let frame_src = format!("frame-src 'self' {}", self.frame_sources.join(" "));
        let csp = [
            "default-src 'self'".to_string(),
            "script-src 'self'".to_string(),
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com".to_string(),
            "font-src 'self' https://fonts.gstatic.com".to_string(),
            "img-src 'self' data:".to_string(),
            format!("connect-src 'self' ws://127.0.0.1:{port} ws://localhost:{port}"),
            frame_src.trim().to_string(),
            "frame-ancestors 'none'".to_string(),
            "base-uri 'self'".to_string(),
            "object-src 'none'".to_string(),
            "form-action 'none'".to_string(),
        ]
        .join("; ");
        vec![
            ("cache-control", "no-store".into()),
            ("content-security-policy", csp),
            ("cross-origin-opener-policy", "same-origin".into()),
            ("cross-origin-resource-policy", "same-origin".into()),
            (
                "permissions-policy",
                "camera=(), microphone=(), geolocation=()".into(),
            ),
            ("referrer-policy", "no-referrer".into()),
            ("x-content-type-options", "nosniff".into()),
            ("x-frame-options", "DENY".into()),
        ]
    }
}

#[cfg(test)]
mod tests {
    //! `field/server/test/security.test.mjs`, the policy half. The real
    //! upgrade boundary is exercised by the server's own tests.
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;

    const PORT: u16 = 7749;

    struct Req {
        method: &'static str,
        host: String,
        origin: Option<String>,
        cookie: Option<&'static str>,
        authorization: Option<String>,
        content_type: Option<&'static str>,
        remote: &'static str,
    }

    impl Default for Req {
        fn default() -> Self {
            Req {
                method: "GET",
                host: format!("127.0.0.1:{PORT}"),
                origin: None,
                cookie: None,
                authorization: None,
                content_type: None,
                remote: "127.0.0.1",
            }
        }
    }

    fn facts(path: &str, bootstrap: Option<&str>, r: Req) -> RequestFacts {
        RequestFacts {
            method: r.method.into(),
            path: path.into(),
            bootstrap: bootstrap.map(String::from),
            host: Some(r.host),
            origin: r.origin,
            cookie: r.cookie.map(String::from),
            authorization: r.authorization,
            content_type: r.content_type.map(String::from),
            remote_address: r.remote.into(),
        }
    }

    fn fixture() -> ControlSecurity {
        let mut options = SecurityOptions::new(PORT);
        options.bootstrap_token = Some("bootstrap-fixture-token".into());
        options.browser_token = Some("browser-fixture-token".into());
        ControlSecurity::new(options).unwrap()
    }

    const COOKIE: &str = "unrelated=1; field_session=browser-fixture-token; another=2";

    fn status(result: Result<Authority, Denied>) -> u16 {
        result.unwrap_err().status
    }

    #[test]
    fn tokens_and_loopback_helpers() {
        assert!(generate_token().len() >= 43);
        assert_ne!(generate_token(), generate_token());
        assert!(is_loopback_address("::ffff:127.0.0.1"));
        assert!(!is_loopback_address("192.168.1.20"));
    }

    #[test]
    fn bootstrap_split_authority_loopback_host_origin_cookie_and_websocket_gates() {
        let mut security = fixture();
        let internal = security.mint_harness_token("session-fixture");

        let boot = security.consume_bootstrap(&facts(
            "/",
            Some("bootstrap-fixture-token"),
            Req::default(),
        ));
        assert_eq!(
            boot,
            Bootstrap::Granted {
                redirect: "/".into(),
                cookie: Some(
                    "field_session=browser-fixture-token; HttpOnly; SameSite=Strict; Path=/".into()
                )
            }
        );

        let state = |r: Req| facts("/api/state", None, r);
        assert!(security
            .authorize_request(&state(Req {
                cookie: Some(COOKIE),
                ..Req::default()
            }))
            .is_ok());
        assert_eq!(
            status(security.authorize_request(&state(Req {
                cookie: Some("field_session=wrong"),
                ..Req::default()
            }))),
            401
        );
        assert_eq!(
            status(security.authorize_request(&state(Req {
                cookie: Some(COOKIE),
                host: "evil.example".into(),
                ..Req::default()
            }))),
            421
        );
        assert_eq!(
            status(security.authorize_request(&state(Req {
                cookie: Some(COOKIE),
                remote: "10.0.0.8",
                ..Req::default()
            }))),
            403
        );

        let command = |r: Req| facts("/api/command", None, r);
        assert_eq!(
            status(security.authorize_request(&command(Req {
                method: "POST",
                cookie: Some(COOKIE),
                ..Req::default()
            }))),
            403
        );
        assert!(security
            .authorize_request(&command(Req {
                method: "POST",
                cookie: Some(COOKIE),
                origin: Some(format!("http://127.0.0.1:{PORT}")),
                ..Req::default()
            }))
            .is_ok());
        assert_eq!(
            status(security.authorize_request(&command(Req {
                method: "POST",
                cookie: Some(COOKIE),
                origin: Some(format!("http://localhost:{PORT}")),
                ..Req::default()
            }))),
            403,
            "localhost is not the same origin as 127.0.0.1"
        );

        let internal_url = |r: Req| facts("/api/internal/permission", None, r);
        assert_eq!(
            security.authorize_request(&internal_url(Req {
                method: "POST",
                authorization: Some(format!("Bearer {internal}")),
                content_type: Some("application/json"),
                ..Req::default()
            })),
            Ok(Authority::Harness {
                session_id: "session-fixture".into()
            })
        );
        assert_eq!(
            status(security.authorize_request(&internal_url(Req {
                method: "POST",
                cookie: Some(COOKIE),
                content_type: Some("application/json"),
                ..Req::default()
            }))),
            401,
            "the operator cookie grants nothing to the harness endpoint"
        );
        assert_eq!(
            status(security.authorize_request(&internal_url(Req {
                method: "POST",
                authorization: Some(format!("Bearer {internal}")),
                content_type: Some("text/plain"),
                ..Req::default()
            }))),
            415
        );

        let ws = |r: Req| facts("/ws", None, r);
        assert!(security
            .authorize_upgrade(&ws(Req {
                cookie: Some(COOKIE),
                origin: Some(format!("http://127.0.0.1:{PORT}")),
                ..Req::default()
            }))
            .is_ok());
        assert_eq!(
            status(security.authorize_upgrade(&ws(Req {
                cookie: Some(COOKIE),
                ..Req::default()
            }))),
            403
        );

        let replay = security.consume_bootstrap(&facts(
            "/",
            Some("bootstrap-fixture-token"),
            Req::default(),
        ));
        assert!(
            matches!(replay, Bootstrap::Denied(Denied { status: 410, .. })),
            "{replay:?}"
        );
        let already = security.consume_bootstrap(&facts(
            "/",
            Some("bootstrap-fixture-token"),
            Req {
                cookie: Some(COOKIE),
                ..Req::default()
            },
        ));
        assert_eq!(
            already,
            Bootstrap::Granted {
                redirect: "/".into(),
                cookie: None
            }
        );

        security.revoke_harness_token("session-fixture");
        assert_eq!(
            status(security.authorize_request(&internal_url(Req {
                method: "POST",
                authorization: Some(format!("Bearer {internal}")),
                content_type: Some("application/json"),
                ..Req::default()
            }))),
            401
        );

        let leftover = security.mint_harness_token("still-running");
        assert!(security.revoke_browser_session().contains("Max-Age=0"));
        security.revoke_all_harness_tokens();
        assert_eq!(
            status(security.authorize_request(&command(Req {
                cookie: Some(COOKIE),
                ..Req::default()
            }))),
            401
        );
        assert_eq!(
            status(security.authorize_upgrade(&ws(Req {
                cookie: Some(COOKIE),
                origin: Some(format!("http://127.0.0.1:{PORT}")),
                ..Req::default()
            }))),
            401
        );
        assert!(matches!(
            security.consume_bootstrap(&facts(
                "/",
                Some("bootstrap-fixture-token"),
                Req::default()
            )),
            Bootstrap::Denied(Denied { status: 410, .. })
        ));
        assert_eq!(
            status(security.authorize_request(&internal_url(Req {
                method: "POST",
                authorization: Some(format!("Bearer {leftover}")),
                content_type: Some("application/json"),
                ..Req::default()
            }))),
            401
        );
    }

    #[test]
    fn a_dev_ui_origin_is_trusted_only_when_exact() {
        let mut options = SecurityOptions::new(PORT);
        options.trusted_origins = vec!["http://127.0.0.1:7748".into()];
        options.bootstrap_redirect = "http://127.0.0.1:7748".into();
        options.bootstrap_token = Some("dev-bootstrap-token".into());
        options.browser_token = Some("dev-browser-token".into());
        let mut dev = ControlSecurity::new(options).unwrap();
        let boot = dev.consume_bootstrap(&facts("/", Some("dev-bootstrap-token"), Req::default()));
        assert!(
            matches!(boot, Bootstrap::Granted { ref redirect, .. } if redirect == "http://127.0.0.1:7748")
        );
        let command = |origin: &str| {
            facts(
                "/api/command",
                None,
                Req {
                    method: "POST",
                    cookie: Some("field_session=dev-browser-token"),
                    origin: Some(origin.into()),
                    ..Req::default()
                },
            )
        };
        assert!(dev
            .authorize_request(&command("http://127.0.0.1:7748"))
            .is_ok());
        assert_eq!(
            status(dev.authorize_request(&command("http://127.0.0.1:7750"))),
            403
        );

        let mut bad = SecurityOptions::new(PORT);
        bad.trusted_origins = vec!["https://evil.example".into()];
        assert!(matches!(
            ControlSecurity::new(bad),
            Err(SecurityError::TrustedOrigin)
        ));
    }

    #[test]
    fn an_expired_bootstrap_and_a_restarted_server_both_refuse() {
        let clock = Arc::new(AtomicI64::new(0));
        let mut options = SecurityOptions::new(PORT);
        options.bootstrap_ttl_ms = 100;
        options.bootstrap_token = Some("expires".into());
        let reader = clock.clone();
        options.now = Some(Box::new(move || reader.load(Ordering::SeqCst)));
        let mut expiring = ControlSecurity::new(options).unwrap();
        clock.store(100, Ordering::SeqCst);
        assert!(matches!(
            expiring.consume_bootstrap(&facts("/", Some("expires"), Req::default())),
            Bootstrap::Denied(Denied { status: 401, .. })
        ));

        let restarted = ControlSecurity::new(SecurityOptions::new(PORT)).unwrap();
        assert_eq!(
            status(restarted.authorize_request(&facts(
                "/api/command",
                None,
                Req {
                    cookie: Some(COOKIE),
                    ..Req::default()
                }
            ))),
            401,
            "a fresh process mints fresh tokens; an old cookie is worthless"
        );
    }

    #[test]
    fn the_headers_pin_the_page_to_its_own_origin() {
        let security = fixture();
        let headers = security.headers();
        let csp = headers
            .iter()
            .find(|(k, _)| *k == "content-security-policy")
            .unwrap()
            .1
            .clone();
        assert!(csp.contains("frame-ancestors 'none'"));
        assert!(csp.contains(&format!("ws://127.0.0.1:{PORT}")));
        assert!(headers
            .iter()
            .any(|(k, v)| *k == "x-frame-options" && v == "DENY"));
    }
}
