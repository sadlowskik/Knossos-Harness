//! What may reach the disk. Port of `sanitizeEventData` in
//! `field/server/src/store/db.js`.
//!
//! An event's payload is whatever an adapter or a tool handed over, and tools
//! echo things: a header, a key, a token in a URL. The log is the durable
//! record of everything that happened, so a credential that lands in it lives
//! forever and travels with every export. Redaction therefore happens at the
//! one gate everything passes through, and it is bounded, because a payload
//! that is allowed to be arbitrarily large is a way to fill the disk.

use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

const MAX_STRING_CHARS: usize = 16_000;
const MAX_TOTAL_CHARS: usize = 256_000;
const MAX_ITEMS: usize = 300;
const MAX_DEPTH: usize = 12;

/// Keys whose value is a credential by construction, whatever it holds.
fn secret_key() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(^|[_-])(password|passwd|secret|token|authorization|cookie|api[_-]?key|private[_-]?key|bearer)($|[_-])",
        )
        .expect("static regex")
    })
}

fn bearer() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b").expect("static regex"))
}

fn sk_token() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bsk-[A-Za-z0-9_-]{8,}\b").expect("static regex"))
}

/// Whether a key names a credential.
pub fn is_secret_key(key: &str) -> bool {
    secret_key().is_match(key)
}

struct Budget<'a> {
    chars: usize,
    items: usize,
    truncated: bool,
    secrets: &'a [String],
}

/// Redact credential-shaped keys and registered secret values, and bound the
/// payload's size, depth and item count. `secrets` shorter than eight
/// characters are ignored by the caller's contract (see `EventLog`).
pub fn sanitize_event_data(value: &Value, secrets: &[String]) -> Value {
    let mut budget = Budget {
        chars: 0,
        items: 0,
        truncated: false,
        secrets,
    };
    sanitize(value, &mut budget, 0)
}

fn sanitize(value: &Value, budget: &mut Budget<'_>, depth: usize) -> Value {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => return value.clone(),
        _ => {}
    }
    // Same short-circuit order as the reference: the item counter only
    // advances when the depth check passed.
    let over = depth > MAX_DEPTH
        || {
            let seen = budget.items;
            budget.items += 1;
            seen >= MAX_ITEMS
        }
        || budget.chars >= MAX_TOTAL_CHARS;
    if over {
        budget.truncated = true;
        return Value::String("[TRUNCATED]".into());
    }
    match value {
        Value::String(text) => Value::String(sanitize_string(text, budget)),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                if budget.items >= MAX_ITEMS || budget.chars >= MAX_TOTAL_CHARS {
                    out.push(Value::String("[TRUNCATED]".into()));
                    budget.truncated = true;
                    break;
                }
                out.push(sanitize(item, budget, depth + 1));
            }
            Value::Array(out)
        }
        Value::Object(fields) => {
            let mut out = Map::with_capacity(fields.len());
            for (key, item) in fields {
                if budget.items >= MAX_ITEMS || budget.chars >= MAX_TOTAL_CHARS {
                    out.insert("_truncated".into(), Value::Bool(true));
                    budget.truncated = true;
                    break;
                }
                let safe = if is_secret_key(key) {
                    Value::String("[REDACTED]".into())
                } else {
                    sanitize(item, budget, depth + 1)
                };
                out.insert(key.clone(), safe);
            }
            Value::Object(out)
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => unreachable!("handled above"),
    }
}

fn sanitize_string(text: &str, budget: &mut Budget<'_>) -> String {
    let mut safe = text.to_string();
    for secret in budget.secrets {
        if !secret.is_empty() {
            safe = safe.replace(secret.as_str(), "[REDACTED]");
        }
    }
    let safe = bearer().replace_all(&safe, "Bearer [REDACTED]");
    let safe = sk_token().replace_all(&safe, "[REDACTED]");
    let len = safe.chars().count();
    let room = MAX_STRING_CHARS.min(MAX_TOTAL_CHARS.saturating_sub(budget.chars));
    budget.chars += len.min(room);
    if len > room {
        budget.truncated = true;
        let mut cut: String = safe.chars().take(room).collect();
        cut.push_str("…[TRUNCATED]");
        return cut;
    }
    safe.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn credential_shaped_keys_are_redacted_and_counters_are_not() {
        let out = sanitize_event_data(
            &json!({"inputTokens": 42, "api_key": "x", "nested": {"authorization": "Bearer y"}, "token_count": 3}),
            &[],
        );
        assert_eq!(out["inputTokens"], 42);
        assert_eq!(out["api_key"], "[REDACTED]");
        assert_eq!(out["nested"]["authorization"], "[REDACTED]");
        assert_eq!(
            out["token_count"], "[REDACTED]",
            "token_ prefix matches the reference regex"
        );
    }

    #[test]
    fn registered_secrets_and_token_shapes_are_scrubbed_from_strings() {
        let secrets = vec!["configured-secret-value-92".to_string()];
        let out = sanitize_event_data(
            &json!("echoed configured-secret-value-92 and sk-examplecredential99 and Bearer abcdefghij"),
            &secrets,
        );
        let text = out.as_str().unwrap();
        assert!(!text.contains("configured-secret-value-92"));
        assert!(!text.contains("sk-examplecredential99"));
        assert!(text.contains("Bearer [REDACTED]"));
    }

    #[test]
    fn a_long_string_is_cut_and_marked() {
        let long = "a".repeat(MAX_STRING_CHARS + 5);
        let out = sanitize_event_data(&json!(long), &[]);
        let text = out.as_str().unwrap();
        assert!(text.ends_with("…[TRUNCATED]"));
        assert_eq!(
            text.chars().count(),
            MAX_STRING_CHARS + "…[TRUNCATED]".chars().count()
        );
    }

    #[test]
    fn too_many_items_truncate_rather_than_grow() {
        let items: Vec<Value> = (0..MAX_ITEMS + 10).map(|i| json!(i.to_string())).collect();
        let out = sanitize_event_data(&Value::Array(items), &[]);
        let arr = out.as_array().unwrap();
        assert!(arr.len() <= MAX_ITEMS + 1);
        assert_eq!(arr.last().unwrap(), "[TRUNCATED]");
    }

    #[test]
    fn depth_beyond_the_limit_is_truncated() {
        let mut v = json!("leaf");
        for _ in 0..(MAX_DEPTH + 2) {
            v = json!({ "n": v });
        }
        let out = sanitize_event_data(&v, &[]);
        let text = out.to_string();
        assert!(text.contains("[TRUNCATED]"));
        assert!(!text.contains("leaf"));
    }
}
