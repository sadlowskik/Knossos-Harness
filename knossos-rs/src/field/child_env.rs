//! The minimal environment a harness child receives. Port of
//! `field/server/src/child-env.js`.
//!
//! A base allowlist, the one provider's own keys, any explicitly requested
//! keys, and per-session overrides. The operator's privileged Cameo secrets
//! never reach a child even when requested; the per-endpoint Cameo *serve*
//! key is the one exception because the harness needs it to authenticate.

use std::collections::BTreeMap;

const BASE_KEYS: [&str; 24] = [
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "HOME",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "COLORTERM",
    "NO_COLOR",
    "FORCE_COLOR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    "CLAUDE_CONFIG_DIR",
];

const PRIVILEGED_NEVER: [&str; 5] = [
    "CAMEO_CONSOLE_KEY",
    "CAMEO_UPDATE_PRIVATE_PEM",
    "CAMEO_UPDATE_PRIVATE",
    "SSH_AUTH_SOCK",
    "SSH_PRIVATE_KEY",
];

const PRIVILEGED_ALLOW: [&str; 1] = ["CAMEO_SERVE_KEY"];

fn provider_keys(provider: Option<&str>) -> &'static [&'static str] {
    match provider {
        Some("anthropic") => &[
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
        ],
        _ => &[],
    }
}

pub fn is_privileged_env_key(key: &str) -> bool {
    let name = key.to_ascii_uppercase();
    if PRIVILEGED_ALLOW.contains(&name.as_str()) {
        return false;
    }
    if PRIVILEGED_NEVER.contains(&name.as_str()) {
        return true;
    }
    name.starts_with("CAMEO_")
        && (name.contains("KEY")
            || name.contains("PRIVATE")
            || name.contains("SECRET")
            || name.contains("TOKEN"))
}

fn source_value<'a>(source: &'a BTreeMap<String, String>, wanted: &str) -> Option<&'a String> {
    if !cfg!(windows) {
        return source.get(wanted);
    }
    source
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(wanted))
        .map(|(_, v)| v)
}

/// Build a minimal environment for a harness and every subprocess it launches.
pub fn build_child_environment(
    source: &BTreeMap<String, String>,
    provider: Option<&str>,
    explicit_keys: &[String],
    overrides: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let mut keys: Vec<String> = BASE_KEYS.iter().map(|k| k.to_string()).collect();
    keys.extend(provider_keys(provider).iter().map(|k| k.to_string()));
    keys.extend(explicit_keys.iter().cloned());
    for key in keys {
        if is_privileged_env_key(&key) {
            continue;
        }
        if let Some(value) = source_value(source, &key).filter(|v| !v.is_empty()) {
            result.insert(key, value.clone());
        }
    }
    for (key, value) in overrides {
        if is_privileged_env_key(key) {
            continue;
        }
        result.insert(key.clone(), value.clone());
    }
    result
}

/// The process environment as the source map.
pub fn process_environment() -> BTreeMap<String, String> {
    std::env::vars().collect()
}

#[cfg(test)]
mod tests {
    //! `field/server/test/child-env.test.mjs`, case for case.
    use super::*;

    fn source() -> BTreeMap<String, String> {
        [
            ("PATH", "/bin"),
            ("USERPROFILE", "C:/fixture"),
            ("ANTHROPIC_API_KEY", "anthropic-canary"),
            ("OPENAI_API_KEY", "openai-canary"),
            ("AWS_SECRET_ACCESS_KEY", "aws-canary"),
            ("GITHUB_TOKEN", "github-canary"),
            ("NPM_TOKEN", "npm-canary"),
            ("SSH_AUTH_SOCK", "ssh-canary"),
            ("CI_JOB_TOKEN", "ci-canary"),
            ("CAMEO_CONSOLE_KEY", "cameo-operator-canary"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn overrides(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn base_allowlist_provider_isolation_explicit_credentials_and_canary_scrubbing() {
        let anthropic = build_child_environment(
            &source(),
            Some("anthropic"),
            &[],
            &overrides(&[("FIELD_SESSION_ID", "session-1")]),
        );
        assert_eq!(anthropic["PATH"], "/bin");
        assert_eq!(anthropic["ANTHROPIC_API_KEY"], "anthropic-canary");
        assert_eq!(anthropic["FIELD_SESSION_ID"], "session-1");
        for secret in [
            "OPENAI_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "NPM_TOKEN",
            "SSH_AUTH_SOCK",
            "CI_JOB_TOKEN",
            "CAMEO_CONSOLE_KEY",
        ] {
            assert!(
                !anthropic.contains_key(secret),
                "{secret} leaked into Anthropic child"
            );
        }

        let cameo =
            build_child_environment(&source(), Some("openai-compatible"), &[], &BTreeMap::new());
        assert!(!cameo.contains_key("ANTHROPIC_API_KEY"));
        assert!(!cameo.contains_key("OPENAI_API_KEY"));

        let explicit = build_child_environment(
            &source(),
            Some("openai-compatible"),
            &["OPENAI_API_KEY".into(), "CAMEO_CONSOLE_KEY".into()],
            &overrides(&[
                ("CAMEO_CONSOLE_KEY", "override-operator-canary"),
                ("FIELD_SESSION_ID", "session-2"),
            ]),
        );
        assert_eq!(explicit["OPENAI_API_KEY"], "openai-canary");
        assert!(!explicit.contains_key("ANTHROPIC_API_KEY"));
        assert!(
            !explicit.contains_key("CAMEO_CONSOLE_KEY"),
            "Cameo operator key must not reach a child even if requested"
        );
        assert_eq!(explicit["FIELD_SESSION_ID"], "session-2");
        assert!(!is_privileged_env_key("CAMEO_SERVE_KEY"));
        assert!(is_privileged_env_key("CAMEO_UPDATE_PRIVATE"));
    }
}
