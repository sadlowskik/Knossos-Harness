//! Heartbeat to a Cameo node's deck (`POST /api/sessions`).
//!
//! Failures are silent: a box without `cameod` is a normal Knossos run.

use serde_json::Value;

fn console_url() -> String {
    std::env::var("CAMEO_CONSOLE_URL")
        .or_else(|_| {
            std::env::var("CAMEO_BASE_URL")
                .map(|u| u.trim_end_matches('/').trim_end_matches("/v1").to_string())
        })
        .unwrap_or_else(|_| "http://127.0.0.1:9090".into())
}

/// Fire-and-forget upsert. Safe to call from ACP's worker thread.
pub fn report(body: Value) {
    let url = format!("{}/api/sessions", console_url());
    let key = std::env::var("CAMEO_CONSOLE_KEY").ok();
    let _ = std::thread::Builder::new()
        .name("cameo-board".into())
        .spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            rt.block_on(async move {
                let mut req = reqwest::Client::new().post(&url).json(&body);
                if let Some(k) = key {
                    req = req.bearer_auth(k);
                }
                let _ = req.send().await;
            });
        });
}
