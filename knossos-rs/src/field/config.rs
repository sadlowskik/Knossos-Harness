//! The git-backed configuration under `field/`. Port of
//! `field/server/src/config.js` and the workspace canonicalization in
//! `workspace-path.js`.
//!
//! Everything here is durable and human-readable; nothing operational is
//! written back. A workspace whose path does not exist is reported as
//! unmounted rather than silently invented.

use super::js::{get, get_arr, get_bool, get_str, Obj};
use super::projection::{
    EndpointConfig, FieldConfig, RoutineConfig, WebsiteConfig, WorkspaceConfig,
};
use regex::Regex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Everything `field.yaml` and the record directories declare.
#[derive(Debug, Clone, Default)]
pub struct FieldSettings {
    pub field_dir: PathBuf,
    pub field: Value,
    pub defaults: Value,
    pub workspaces: Vec<Value>,
    pub endpoints: Vec<Value>,
    pub websites: Vec<Value>,
    pub roles: Vec<Value>,
    pub agents: Vec<Value>,
    pub missions: Vec<Value>,
    pub constitutions: Vec<Value>,
    pub skills: Vec<Value>,
    pub routines: Vec<Value>,
    pub memory: Vec<Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("field.yaml not found at {0}")]
    Missing(PathBuf),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("field.yaml is not valid YAML: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    #[error("field.yaml could not be represented as JSON: {0}")]
    Json(#[from] serde_json::Error),
}

fn yaml_to_json(text: &str) -> Result<Value, ConfigError> {
    let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)?;
    Ok(serde_json::to_value(parsed)?)
}

/// Split `---\nyaml\n---\nbody` into the data and the body.
pub fn frontmatter(text: &str) -> (Value, String) {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?s)^---\r?\n(.*?)\r?\n---\r?\n?(.*)$").expect("static regex")
    });
    match re.captures(text) {
        Some(caps) => {
            let data = yaml_to_json(&caps[1])
                .ok()
                .filter(|v| !v.is_null())
                .unwrap_or_else(|| json!({}));
            (data, caps[2].to_string())
        }
        None => (json!({}), text.to_string()),
    }
}

fn read_dir(dir: &Path, ext: &str) -> Vec<Value> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|f| f.ends_with(ext))
        .collect();
    names.sort();
    names
        .into_iter()
        .filter_map(|f| {
            let full = dir.join(&f);
            let text = std::fs::read_to_string(&full).ok()?;
            let stem = f.trim_end_matches(ext).to_string();
            let mut record: Obj;
            if ext == ".md" {
                let (data, body) = frontmatter(&text);
                record = data.as_object().cloned().unwrap_or_default();
                let id = get(&data, "id")
                    .or(get(&data, "name"))
                    .cloned()
                    .unwrap_or(json!(stem));
                record.insert("id".into(), id);
                record.insert("body".into(), json!(body));
            } else {
                let data = yaml_to_json(&text)
                    .ok()
                    .filter(|v| !v.is_null())
                    .unwrap_or_else(|| json!({}));
                record = data.as_object().cloned().unwrap_or_default();
                let id = get(&data, "id").cloned().unwrap_or(json!(stem));
                record.insert("id".into(), id);
            }
            record.insert("file".into(), json!(full.to_string_lossy()));
            Some(Value::Object(record))
        })
        .collect()
}

/// Resolve a workspace against the real filesystem.
pub fn canonicalize_workspace(workspace: &Value, field_dir: &Path) -> Value {
    let requested = field_dir.join(get_str(workspace, "path").unwrap_or("."));
    let requested = normalize(&requested);
    let canonical = std::fs::metadata(&requested)
        .ok()
        .filter(|m| m.is_dir())
        .and_then(|_| dunce_canonicalize(&requested));
    let mounted = canonical.is_some();
    let root = canonical.clone().unwrap_or(requested);
    let mut out = workspace.as_object().cloned().unwrap_or_default();
    out.insert("path".into(), json!(root.to_string_lossy()));
    out.insert(
        "canonicalPath".into(),
        canonical
            .map(|p| json!(p.to_string_lossy()))
            .unwrap_or(Value::Null),
    );
    out.insert("mounted".into(), Value::Bool(mounted));
    Value::Object(out)
}

/// `path.resolve`: absolute, with `.` and `..` folded, without touching disk.
fn normalize(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `fs.realpathSync.native` without the `\\?\` prefix Windows adds.
pub fn dunce_canonicalize(path: &Path) -> Option<PathBuf> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let text = canonical.to_string_lossy();
    Some(match text.strip_prefix(r"\\?\") {
        Some(stripped) if !stripped.starts_with("UNC") => PathBuf::from(stripped),
        _ => canonical,
    })
}

pub fn load_config(field_dir: &Path) -> Result<FieldSettings, ConfigError> {
    let root_file = field_dir.join("field.yaml");
    if !root_file.is_file() {
        return Err(ConfigError::Missing(root_file));
    }
    let root = yaml_to_json(&std::fs::read_to_string(&root_file)?)?;
    let list = |key: &str| get_arr(&root, key).cloned().unwrap_or_default();
    let workspaces = list("workspaces")
        .iter()
        .map(|w| canonicalize_workspace(w, field_dir))
        .collect();
    Ok(FieldSettings {
        field_dir: field_dir.to_path_buf(),
        field: get(&root, "field").cloned().unwrap_or_else(|| json!({})),
        defaults: get(&root, "defaults").cloned().unwrap_or_else(|| json!({})),
        workspaces,
        endpoints: list("endpoints"),
        websites: list("websites"),
        roles: read_dir(&field_dir.join("roles"), ".md"),
        agents: read_dir(&field_dir.join("agents"), ".md"),
        missions: read_dir(&field_dir.join("missions"), ".md"),
        constitutions: read_dir(&field_dir.join("constitutions"), ".md"),
        skills: read_dir(&field_dir.join("skills"), ".md"),
        routines: read_dir(&field_dir.join("routines"), ".yaml"),
        memory: read_dir(&field_dir.join("memory"), ".md"),
    })
}

impl FieldSettings {
    /// What the projection seeds itself from.
    pub fn projection_config(&self) -> FieldConfig {
        FieldConfig {
            workspaces: self
                .workspaces
                .iter()
                .map(|w| WorkspaceConfig {
                    id: get_str(w, "id").unwrap_or("").to_string(),
                    name: get_str(w, "name").map(str::to_string),
                    path: get_str(w, "path").map(str::to_string),
                    mounted: get_bool(w, "mounted"),
                    region: get(w, "region").cloned(),
                })
                .collect(),
            endpoints: self
                .endpoints
                .iter()
                .map(|e| EndpointConfig {
                    id: get_str(e, "id").unwrap_or("").to_string(),
                    name: get_str(e, "name").map(str::to_string),
                    kind: get_str(e, "kind").map(str::to_string),
                    model: get_str(e, "model").map(str::to_string),
                    base_url: get_str(e, "base_url").map(str::to_string),
                    cost_per_mtok: get(e, "cost_per_mtok").cloned(),
                })
                .collect(),
            websites: self
                .websites
                .iter()
                .map(|s| WebsiteConfig {
                    domain: get_str(s, "domain").unwrap_or("").to_string(),
                    label: get_str(s, "label").map(str::to_string),
                })
                .collect(),
            routines: self
                .routines
                .iter()
                .map(|r| RoutineConfig {
                    id: get_str(r, "id").unwrap_or("").to_string(),
                    enabled: get_bool(r, "enabled").unwrap_or(false),
                })
                .collect(),
        }
    }

    /// The `GET /api/config` payload.
    pub fn view(&self) -> Value {
        let strip = |o: &Value| -> Value {
            let mut out = o.as_object().cloned().unwrap_or_default();
            out.remove("body");
            out.remove("file");
            Value::Object(out)
        };
        json!({
            "field": self.field,
            "defaults": self.defaults,
            "workspaces": self.workspaces.iter().map(|w| json!({
                "id": w.get("id"), "name": w.get("name"), "path": w.get("path"), "mounted": w.get("mounted"), "region": w.get("region"),
            })).collect::<Vec<_>>(),
            "endpoints": self.endpoints,
            "websites": self.websites,
            "roles": self.roles.iter().map(strip).collect::<Vec<_>>(),
            "agents": self.agents.iter().map(strip).collect::<Vec<_>>(),
            "missions": self.missions.iter().map(|m| {
                let mut out = strip(m);
                if let (Value::Object(o), Some(body)) = (&mut out, m.get("body")) {
                    o.insert("body".into(), body.clone());
                }
                out
            }).collect::<Vec<_>>(),
            "routines": self.routines,
            "skills": self.skills.iter().map(strip).collect::<Vec<_>>(),
            "memory": self.memory.iter().map(strip).collect::<Vec<_>>(),
            "constitutions": self.constitutions.iter().map(strip).collect::<Vec<_>>(),
        })
    }

    pub fn api_port(&self) -> Option<u16> {
        get(&self.field, "api_port")
            .and_then(Value::as_u64)
            .and_then(|p| u16::try_from(p).ok())
    }

    pub fn workspace(&self, id: &str) -> Option<&Value> {
        self.workspaces
            .iter()
            .find(|w| get_str(w, "id") == Some(id))
    }

    pub fn agents_dir(&self) -> PathBuf {
        self.field_dir.join("agents")
    }

    /// Re-read `field/agents/*.md` so a record written a moment ago is what
    /// the next spawn sees, without a restart.
    pub fn reload_agents(&mut self) {
        self.agents = read_dir(&self.agents_dir(), ".md");
    }

    pub fn agent(&self, id: &str) -> Option<&Value> {
        self.agents.iter().find(|a| get_str(a, "id") == Some(id))
    }

    pub fn has_role(&self, id: &str) -> bool {
        self.roles.iter().any(|r| get_str(r, "id") == Some(id))
    }

    pub fn has_endpoint(&self, id: &str) -> bool {
        self.endpoints.iter().any(|e| get_str(e, "id") == Some(id))
    }
}

/// An agent as `GET /api/agents` reports it: the operator-facing fields and
/// the standing orders (the markdown body), never the raw frontmatter.
pub fn agent_view(agent: &Value) -> Value {
    let id = get_str(agent, "id").unwrap_or("");
    let text = |key: &str| {
        get(agent, key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    json!({
        "id": id,
        "name": text("name").unwrap_or(id),
        "role": text("role"),
        "endpoint": text("endpoint"),
        "model": text("model"),
        "thinking": text("thinking"),
        "orders": get_str(agent, "body").unwrap_or("").trim(),
        "file": agent.get("file"),
    })
}

/// A file-safe id from a display name: lowercase ASCII letters and digits
/// with single dashes between words. Empty when the name has neither.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The fields an agent definition carries, in the order they are written.
#[derive(Debug, Clone, Default)]
pub struct AgentDefinition {
    pub id: String,
    pub name: String,
    pub role: String,
    pub endpoint: String,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub orders: String,
}

/// Write `agents/<id>.md`: `id, name, role, endpoint, model?, thinking?` on
/// top of any other frontmatter the file already had (tool allowances, a
/// constitution), then the orders as the body.
pub fn write_agent_file(
    file: &Path,
    def: &AgentDefinition,
    existing: Option<&Value>,
) -> Result<(), ConfigError> {
    let mut keep: Obj = existing
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for key in [
        "id", "name", "role", "endpoint", "model", "thinking", "body", "file",
    ] {
        keep.remove(key);
    }
    let mut head: Vec<(&str, Value)> = vec![
        ("id", json!(def.id)),
        ("name", json!(def.name)),
        ("role", json!(def.role)),
        ("endpoint", json!(def.endpoint)),
    ];
    if let Some(model) = &def.model {
        head.push(("model", json!(model)));
    }
    if let Some(thinking) = &def.thinking {
        head.push(("thinking", json!(thinking)));
    }
    let mut yaml = String::new();
    for (key, value) in head {
        let line = serde_yaml_ng::to_string(&json!({ key: value })).map_err(ConfigError::Yaml)?;
        yaml.push_str(line.trim_end());
        yaml.push('\n');
    }
    if !keep.is_empty() {
        let rest = serde_yaml_ng::to_string(&Value::Object(keep)).map_err(ConfigError::Yaml)?;
        yaml.push_str(rest.trim_end());
        yaml.push('\n');
    }
    let orders = def.orders.trim();
    let body = if orders.is_empty() {
        String::new()
    } else {
        format!("{orders}\n")
    };
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(file, format!("---\n{yaml}---\n{body}"))?;
    Ok(())
}

/// Update frontmatter fields while preserving the record body verbatim.
pub fn update_frontmatter_file(file: &Path, updates: &Value) -> Result<Value, ConfigError> {
    let text = std::fs::read_to_string(file)?;
    let (data, body) = frontmatter(&text);
    let mut next = data.as_object().cloned().unwrap_or_default();
    if let Value::Object(u) = updates {
        for (k, v) in u {
            next.insert(k.clone(), v.clone());
        }
    }
    let next = Value::Object(next);
    let yaml = serde_yaml_ng::to_string(&next).map_err(ConfigError::Yaml)?;
    let yaml = yaml.trim_end();
    let separator = if body.starts_with('\n') || body.is_empty() {
        ""
    } else {
        "\n"
    };
    std::fs::write(file, format!("---\n{yaml}\n---\n{separator}{body}"))?;
    Ok(next)
}

/// Compose the system prompt an agent runs under: constitution, role, agent
/// body, mission, orders, and the Field reporting protocol.
pub fn compose_prompt(
    cfg: &FieldSettings,
    agent_id: &str,
    role_id: Option<&str>,
    mission_id: Option<&str>,
    orders: Option<&str>,
) -> String {
    let find =
        |list: &[Value], id: &str| list.iter().find(|x| get_str(x, "id") == Some(id)).cloned();
    let agent = find(&cfg.agents, agent_id);
    let role_id = role_id.map(str::to_string).or_else(|| {
        agent
            .as_ref()
            .and_then(|a| get_str(a, "role").map(str::to_string))
    });
    let role = role_id.as_deref().and_then(|r| find(&cfg.roles, r));
    let constitution_id = agent
        .as_ref()
        .and_then(|a| get_str(a, "constitution").map(str::to_string))
        .unwrap_or_else(|| "core".into());
    let constitution = find(&cfg.constitutions, &constitution_id);
    let mission = mission_id.and_then(|m| find(&cfg.missions, m));
    let body = |v: &Value| get_str(v, "body").unwrap_or("").trim().to_string();
    let mut parts: Vec<String> = Vec::new();
    if let Some(c) = &constitution {
        parts.push(body(c));
    }
    if let Some(r) = &role {
        parts.push(body(r));
    }
    if let Some(a) = agent.as_ref().filter(|a| !body(a).is_empty()) {
        parts.push(body(a));
    }
    if let Some(m) = &mission {
        parts.push(format!(
            "# Active mission: {}\n\n{}",
            get_str(m, "name").unwrap_or(""),
            body(m)
        ));
    }
    if let Some(o) = orders.map(str::trim).filter(|o| !o.is_empty()) {
        parts.push(format!("# Orders from the operator\n\n{o}"));
    }
    parts.push(
        "# Field reporting protocol\n\n\
         You are running as a unit inside Field, an operator-controlled multi-agent environment.\n\
         The operator watches your tool calls live. Keep prose short; the work is the output.\n\
         When you finish, state plainly what changed, what you verified, and what you did not."
            .to_string(),
    );
    parts.join("\n\n---\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_splits_data_from_body_and_tolerates_its_absence() {
        let (data, body) = frontmatter("---\nid: rhea\nrole: builder\n---\nBe careful.\n");
        assert_eq!(data["id"], "rhea");
        assert_eq!(body, "Be careful.\n");
        let (data, body) = frontmatter("plain text");
        assert_eq!(data, json!({}));
        assert_eq!(body, "plain text");
    }

    #[test]
    fn slug_is_lowercase_dashed_ascii() {
        assert_eq!(slug("Rhea Coder"), "rhea-coder");
        assert_eq!(slug("  Qwen: 7B / fast!  "), "qwen-7b-fast");
        assert_eq!(slug("---"), "");
        assert_eq!(slug("Ünïcode Náme"), "n-code-n-me");
    }

    #[test]
    fn write_agent_file_keeps_field_order_and_foreign_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("agents").join("rhea.md");
        let def = AgentDefinition {
            id: "rhea".into(),
            name: "Rhea: Coder".into(),
            role: "builder".into(),
            endpoint: "local".into(),
            model: Some("Qwen/Qwen2.5-Coder-7B-Instruct".into()),
            thinking: None,
            orders: "  Keep diffs small.\n".into(),
        };
        write_agent_file(&file, &def, None).unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        assert_eq!(
            text,
            "---\nid: rhea\nname: 'Rhea: Coder'\nrole: builder\nendpoint: local\nmodel: Qwen/Qwen2.5-Coder-7B-Instruct\n---\nKeep diffs small.\n"
        );
        let existing = json!({ "id": "rhea", "tools_allow": ["Read"], "body": "old", "file": "x" });
        let mut next = def.clone();
        next.model = None;
        next.thinking = Some("high".into());
        next.orders = String::new();
        write_agent_file(&file, &next, Some(&existing)).unwrap();
        let (data, body) = frontmatter(&std::fs::read_to_string(&file).unwrap());
        assert_eq!(data["thinking"], "high");
        assert!(data.get("model").is_none());
        assert_eq!(data["tools_allow"], json!(["Read"]));
        assert_eq!(body, "");
        assert_eq!(
            agent_view(&json!({ "id": "x", "body": " hi \n" }))["orders"],
            "hi"
        );
    }

    #[test]
    fn a_missing_workspace_is_unmounted_not_invented() {
        let dir = tempfile::tempdir().unwrap();
        let ws = canonicalize_workspace(
            &json!({ "id": "ghost", "path": "does-not-exist" }),
            dir.path(),
        );
        assert_eq!(ws["mounted"], false);
        assert!(ws["canonicalPath"].is_null());
        let real = canonicalize_workspace(&json!({ "id": "here", "path": "." }), dir.path());
        assert_eq!(real["mounted"], true);
    }

    #[test]
    fn load_config_reads_records_and_seeds_the_projection() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("field.yaml"),
            "field:\n  name: Test\n  api_port: 7751\nworkspaces:\n  - id: here\n    name: Here\n    path: .\nendpoints:\n  - id: local\n    kind: openai-compatible\n    model: m\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("roles")).unwrap();
        std::fs::write(
            dir.path().join("roles").join("builder.md"),
            "---\nname: builder\n---\nBuild things.\n",
        )
        .unwrap();
        let settings = load_config(dir.path()).unwrap();
        assert_eq!(settings.api_port(), Some(7751));
        assert_eq!(settings.roles[0]["id"], "builder");
        assert_eq!(settings.roles[0]["body"], "Build things.\n");
        let view = settings.view();
        assert!(
            view["roles"][0].get("body").is_none(),
            "roles are stripped of their body"
        );
        assert_eq!(view["workspaces"][0]["mounted"], true);
        let seed = settings.projection_config();
        assert_eq!(seed.workspaces[0].id, "here");
        assert_eq!(seed.endpoints[0].model.as_deref(), Some("m"));
    }
}
