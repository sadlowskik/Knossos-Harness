//! The declarative adapter manifest registry. Port of
//! `field/server/src/harness/adapters.js`.
//!
//! The registry keys on this table to select a harness adapter for an
//! endpoint kind, replacing the old inline `openai-compatible ? Knossos :
//! Claude` ternary with the same mapping expressed as data. The manifests are
//! data only; the concrete adapters live in `claude_session` and
//! `knossos_session`, and the registry maps a manifest id to a constructor.

use serde_json::{json, Value};

use super::adapter::HARNESS_KINDS;

/// Descriptive launch spec: the binary (`ENV|fallback`) and its argv shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub bin: &'static str,
    pub argv: &'static str,
}

/// How credentials reach the child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Auth {
    pub via: &'static str,
    pub keys: &'static [&'static str],
}

/// The canonical `capabilities()` shape every adapter reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterCapabilities {
    pub kind: &'static str,
    pub duplex: bool,
    pub resumable: bool,
    pub permissions: &'static str,
    pub delegation: bool,
    pub browser: bool,
    pub verification: bool,
    pub dry_run: bool,
}

impl AdapterCapabilities {
    pub fn to_value(self) -> Value {
        json!({
            "kind": self.kind,
            "duplex": self.duplex,
            "resumable": self.resumable,
            "permissions": self.permissions,
            "delegation": self.delegation,
            "browser": self.browser,
            "verification": self.verification,
            "dryRun": self.dry_run,
        })
    }
}

/// One adapter manifest: the seam the registry keys on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Manifest {
    /// Stable adapter id, e.g. `claude-code`.
    pub id: &'static str,
    pub name: &'static str,
    /// One of [`HARNESS_KINDS`].
    pub kind: &'static str,
    /// Endpoint/engine kinds that route to this adapter.
    pub endpoint_kinds: &'static [&'static str],
    /// Chosen when no `endpoint_kinds` match.
    pub default: bool,
    pub launch: Option<Launch>,
    pub auth: Option<Auth>,
    pub capabilities: AdapterCapabilities,
}

impl Manifest {
    pub fn capabilities(&self) -> Value {
        self.capabilities.to_value()
    }

    /// The manifest as the Node `defineAdapterManifest` output (minus the
    /// constructor), for operators and conformance checks.
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "endpointKinds": self.endpoint_kinds,
            "launch": self.launch.map(|l| json!({ "bin": l.bin, "argv": l.argv })),
            "endpoint": Value::Null,
            "auth": self.auth.map(|a| json!({ "via": a.via, "keys": a.keys })),
            "capabilities": self.capabilities(),
            "default": self.default,
        })
    }
}

pub const CLAUDE_CODE: &str = "claude-code";
pub const KNOSSOS: &str = "knossos";
pub const ACP: &str = "acp";

pub static ADAPTER_MANIFESTS: [Manifest; 3] = [
    Manifest {
        id: CLAUDE_CODE,
        name: "Claude Code (stream-json duplex)",
        kind: "cli-stream-json",
        // Any endpoint kind that is not openai-compatible falls through to
        // this default, preserving the historical selection behavior exactly.
        endpoint_kinds: &["anthropic"],
        default: true,
        launch: Some(Launch {
            bin: "FIELD_CLAUDE_BIN|claude",
            argv: "-p --output-format stream-json --input-format stream-json --verbose",
        }),
        auth: Some(Auth {
            via: "child-env",
            keys: &["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
        }),
        capabilities: AdapterCapabilities {
            kind: "cli-stream-json",
            duplex: true,
            resumable: true,
            permissions: "mcp-bridge",
            delegation: true,
            browser: true,
            verification: false,
            dry_run: false,
        },
    },
    Manifest {
        id: KNOSSOS,
        name: "Knossos serve (NDJSON)",
        kind: "ndjson-serve",
        endpoint_kinds: &["openai-compatible"],
        default: false,
        launch: Some(Launch {
            bin: "FIELD_KNOSSOS_BIN|knossos",
            argv: "serve --workspace <cwd> --engine <engine>",
        }),
        auth: Some(Auth {
            via: "child-env",
            keys: &["CAMEO_SERVE_KEY", "CAMEO_BASE_URL", "CAMEO_MODEL"],
        }),
        capabilities: AdapterCapabilities {
            kind: "ndjson-serve",
            duplex: true,
            resumable: true,
            permissions: "inline-handshake",
            delegation: false,
            browser: false,
            verification: true,
            dry_run: true,
        },
    },
    Manifest {
        id: ACP,
        name: "Agent Client Protocol (JSON-RPC over stdio)",
        kind: "acp",
        // Intentionally empty: ACP is a harness-kind axis, not an endpoint
        // kind. It is reached only by an explicit harness selector.
        endpoint_kinds: &[],
        default: false,
        launch: Some(Launch {
            bin: "FIELD_ACP_BIN",
            argv: "FIELD_ACP_ARGS",
        }),
        auth: Some(Auth {
            via: "child-env",
            keys: &[],
        }),
        capabilities: AdapterCapabilities {
            kind: "acp",
            duplex: true,
            resumable: true,
            permissions: "inline-handshake",
            delegation: false,
            browser: false,
            verification: false,
            dry_run: true,
        },
    },
];

/// Resolve the adapter manifest for an endpoint/engine kind. A kind no
/// manifest claims falls back to the default manifest, identical to the old
/// ternary where every non `openai-compatible` endpoint used the direct
/// Claude adapter.
///
/// `explicit_harness` is an additive second axis: when an endpoint names a
/// harness by id or kind (e.g. `acp`), that manifest is selected regardless
/// of endpoint kind. Absent, resolution is byte-identical to before.
pub fn select_manifest(
    endpoint_kind: Option<&str>,
    explicit_harness: Option<&str>,
) -> &'static Manifest {
    if let Some(explicit) = explicit_harness.filter(|h| !h.is_empty()) {
        if let Some(chosen) = ADAPTER_MANIFESTS
            .iter()
            .find(|m| m.id == explicit || m.kind == explicit)
        {
            return chosen;
        }
    }
    endpoint_kind
        .and_then(|kind| {
            ADAPTER_MANIFESTS
                .iter()
                .find(|m| m.endpoint_kinds.contains(&kind))
        })
        .or_else(|| ADAPTER_MANIFESTS.iter().find(|m| m.default))
        .expect("one default adapter manifest")
}

pub fn manifest_by_id(id: &str) -> Option<&'static Manifest> {
    ADAPTER_MANIFESTS.iter().find(|m| m.id == id)
}

/// The manifest invariants `defineAdapterManifest` enforced at load time.
pub fn validate_manifests() -> Result<(), String> {
    for m in &ADAPTER_MANIFESTS {
        if m.id.is_empty() {
            return Err("adapter manifest requires a string id".into());
        }
        if !HARNESS_KINDS.contains(&m.kind) {
            return Err(format!(
                "adapter manifest {} has unknown harness kind: {}",
                m.id, m.kind
            ));
        }
        if m.capabilities.kind != m.kind {
            return Err(format!(
                "adapter manifest {} capabilities disagree with its kind",
                m.id
            ));
        }
    }
    let defaults = ADAPTER_MANIFESTS.iter().filter(|m| m.default).count();
    if defaults != 1 {
        return Err(format!(
            "expected one default adapter manifest, found {defaults}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The manifest half of `field/server/test/adapter-contract.test.mjs`.
    use super::*;

    #[test]
    fn manifest_selection_reproduces_the_historical_ternary() {
        assert_eq!(select_manifest(Some("openai-compatible"), None).id, KNOSSOS);
        assert_eq!(select_manifest(Some("anthropic"), None).id, CLAUDE_CODE);
        assert_eq!(
            select_manifest(None, None).id,
            CLAUDE_CODE,
            "unknown kind falls back to the default adapter"
        );
        assert_eq!(
            select_manifest(Some("some-future-kind"), None).id,
            CLAUDE_CODE
        );
        // The explicit harness axis is additive and opt-in.
        assert_eq!(select_manifest(Some("anthropic"), Some("acp")).id, ACP);
        assert_eq!(
            select_manifest(Some("openai-compatible"), Some("cli-stream-json")).id,
            CLAUDE_CODE
        );
        assert_eq!(
            select_manifest(Some("openai-compatible"), Some("no-such-harness")).id,
            KNOSSOS,
            "an unknown explicit harness falls through to endpoint-kind resolution"
        );
    }

    #[test]
    fn manifests_are_well_formed() {
        validate_manifests().unwrap();
        for m in &ADAPTER_MANIFESTS {
            assert!(
                HARNESS_KINDS.contains(&m.kind),
                "{} declares a known harness kind",
                m.id
            );
            assert_eq!(m.capabilities()["kind"], m.kind);
            let v = m.to_value();
            assert_eq!(v["id"], m.id);
            assert!(v["endpointKinds"].is_array());
        }
        assert_eq!(ADAPTER_MANIFESTS.iter().filter(|m| m.default).count(), 1);
        let claude = manifest_by_id(CLAUDE_CODE).unwrap();
        assert_eq!(claude.launch.unwrap().bin, "FIELD_CLAUDE_BIN|claude");
        assert_eq!(
            claude.auth.unwrap().keys,
            &["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
        );
        assert_eq!(claude.capabilities()["permissions"], "mcp-bridge");
        assert!(manifest_by_id("nope").is_none());
    }
}
