//! Permission policy: the hard denials a tool request meets before it ever
//! reaches a human. Port of `denyListFor`, `evaluatePermissionPolicy`,
//! `validateEgressUrl` and `parseCampaignReport` in
//! `field/server/src/harness/registry.js`, plus the glob matcher in
//! `field/server/src/glob.js`.
//!
//! Returning `None` means the request may proceed to the operator's approval
//! queue; it is never auto-allowed here.

use super::js::{get, get_arr, js_string, truthy};
use regex::Regex;
use serde_json::Value;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// Compile one glob into an anchored regex. `/` matches either separator.
pub fn glob_to_regex(glob: &str) -> Regex {
    let chars: Vec<char> = glob.chars().collect();
    let mut out = String::from("^");
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '*' {
            if chars.get(i + 1) == Some(&'*') {
                if chars.get(i + 2) == Some(&'/') {
                    out.push_str(r"(?:.*[\\/])?");
                    i += 3;
                } else {
                    out.push_str(".*");
                    i += 2;
                }
            } else {
                out.push_str(r"[^\\/]*");
                i += 1;
            }
            continue;
        }
        match c {
            '?' => out.push_str(r"[^\\/]"),
            '/' => out.push_str(r"[\\/]"),
            '.' | '+' | '^' | '$' | '{' | '}' | '(' | ')' | '|' | '[' | ']' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            other => out.push(other),
        }
        i += 1;
    }
    out.push('$');
    Regex::new(&out).unwrap_or_else(|_| Regex::new("^$").expect("empty regex"))
}

/// A matcher over a list of globs. A `.../**` pattern also matches the
/// directory node itself, so a watcher does not descend into a tree it was
/// told to skip.
pub fn build_matcher(patterns: &[String]) -> impl Fn(&str) -> bool {
    let mut res = Vec::new();
    for p in patterns {
        res.push(glob_to_regex(p));
        if let Some(stem) = p.strip_suffix("/**") {
            res.push(glob_to_regex(stem));
        }
    }
    move |candidate: &str| res.iter().any(|r| r.is_match(candidate))
}

/// Tools that can change the world. A read-only role is denied these
/// outright, so the declaration in `field/roles` is a real constraint.
const DIRECT_MUTATING_TOOLS: [&str; 4] = ["Edit", "Write", "NotebookEdit", "PowerShell"];
const UNMANAGED_DELEGATION_TOOLS: [&str; 2] = ["Task", "Agent"];

/// The CLI's hard deny list for a role and an optional per-agent loadout.
/// `None` when nothing is denied.
pub fn deny_list_for(role: Option<&Value>, agent: Option<&Value>) -> Option<Vec<String>> {
    let mut denied: Vec<String> = Vec::new();
    let mut add = |tool: String| {
        if !denied.contains(&tool) {
            denied.push(tool);
        }
    };
    for tool in role
        .and_then(|r| get_arr(r, "tools_deny"))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
    {
        add(js_string(tool));
    }
    if role.is_some_and(|r| r.get("read_only").is_some_and(truthy)) {
        for tool in DIRECT_MUTATING_TOOLS {
            add(tool.to_string());
        }
    }
    if let Some(equipped) = agent.and_then(|a| get_arr(a, "tools_allow")) {
        let equipped: Vec<String> = equipped.iter().map(js_string).collect();
        for tool in role
            .and_then(|r| get_arr(r, "tools_allow"))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
        {
            let tool = js_string(tool);
            if !equipped.contains(&tool) {
                add(tool);
            }
        }
    }
    for tool in UNMANAGED_DELEGATION_TOOLS {
        add(tool.to_string());
    }
    if denied.is_empty() {
        None
    } else {
        Some(denied)
    }
}

fn tool_category(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let known = match lower.as_str() {
        "read" | "read_file" => "read",
        "grep" | "search" | "search_code" => "search",
        "glob" | "list_dir" => "list",
        "edit" | "edit_file" => "edit",
        "write" | "write_file" | "notebookedit" => "write",
        "bash" | "powershell" | "shell" | "exec" | "run" | "run_command" => "shell",
        "webfetch" => "web_fetch",
        "websearch" => "web_search",
        "task" | "agent" => "delegate",
        "verify" => "verify",
        "ask_user" | "askuserquestion" => "ask",
        _ => return format!("unknown:{lower}"),
    };
    known.to_string()
}

fn role_tool_category(tool: &str) -> String {
    let lower = tool.to_ascii_lowercase();
    let known = match lower.as_str() {
        "read" => "read",
        "grep" => "search",
        "glob" => "list",
        "edit" => "edit",
        "write" | "notebookedit" => "write",
        "bash" | "powershell" => "shell",
        "webfetch" => "web_fetch",
        "websearch" => "web_search",
        "task" | "agent" => "delegate",
        _ => return format!("exact:{lower}"),
    };
    known.to_string()
}

fn mutating_shell() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(^|[\s;&|])(rm|rmdir|del|erase|move|mv|copy|cp|mkdir|md|touch|tee|set-content|add-content|out-file|remove-item|move-item|copy-item|new-item|git\s+(add|commit|reset|checkout|restore|clean)|npm\s+(install|uninstall)|pnpm\s+(add|remove)|yarn\s+(add|remove)|pip\s+install|cargo\s+(install|fmt|fix))([\s;&|]|$)|(^|[^>])>{1,2}($|[^>])")
            .expect("static regex")
    })
}

fn network_shell() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(^|[\s;&|])(curl|wget|aria2c|fetch|invoke-webrequest|invoke-restmethod|iwr|irm|ssh|scp|sftp|ftp|telnet|nc|ncat|netcat|git\s+(clone|fetch|pull)|npm\s+(install|add)|pnpm\s+(install|add)|yarn\s+(install|add)|pip\s+install|cargo\s+install)([\s;&|]|$)")
            .expect("static regex")
    })
}

fn escapes_scope() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(^|[\s"'`=])\.\.([\\/]|$)"#).expect("static regex"))
}

/// A hard denial, with the message the harness relays to the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denial {
    pub decision: &'static str,
    pub message: String,
}

fn deny(message: impl Into<String>) -> Option<Denial> {
    Some(Denial {
        decision: "deny",
        message: message.into(),
    })
}

/// Validates a workspace-relative write path against the sensitive-file
/// policy; `Err` denies.
pub type WritePathValidator<'a> = &'a dyn Fn(&str) -> Result<(), String>;

/// What the policy needs to know about one request.
#[derive(Default)]
pub struct PolicyInput<'a> {
    pub tool_name: &'a str,
    pub input: Value,
    pub workspace_path: Option<&'a Path>,
    pub read_only: bool,
    pub environment_scope: Option<&'a str>,
    pub allowed_tools: Option<Vec<String>>,
    pub write_scope: Option<Vec<String>>,
    pub validate_write_path: Option<WritePathValidator<'a>>,
    pub allowed_domains: Vec<String>,
}

/// `path.resolve(root, candidate)` without touching disk.
fn resolve_under(root: &Path, candidate: &str) -> PathBuf {
    let joined = if Path::new(candidate).is_absolute() {
        PathBuf::from(candidate)
    } else {
        root.join(candidate)
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn path_key(p: &Path) -> String {
    let text = p
        .to_string_lossy()
        .replace('/', std::path::MAIN_SEPARATOR_STR);
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

fn inside(root: &Path, resolved: &Path) -> bool {
    let (r, c) = (path_key(root), path_key(resolved));
    c == r || c.starts_with(&format!("{r}{}", std::path::MAIN_SEPARATOR))
}

fn relative_to(root: &Path, resolved: &Path) -> String {
    match resolved.strip_prefix(root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => {
            // Outside the root: express it as `../...` the way `path.relative` does.
            let mut ups = 0;
            let mut base = root.to_path_buf();
            while !resolved.starts_with(&base) {
                if !base.pop() {
                    return resolved.to_string_lossy().replace('\\', "/");
                }
                ups += 1;
            }
            let rest = resolved
                .strip_prefix(&base)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            let mut out = "../".repeat(ups);
            out.push_str(&rest);
            out.trim_end_matches('/').to_string()
        }
    }
}

pub fn evaluate_permission_policy(p: &PolicyInput<'_>) -> Option<Denial> {
    let name = p.tool_name;
    let category = tool_category(name);
    let mutating = matches!(category.as_str(), "edit" | "write" | "shell");
    let candidate = ["file_path", "path", "notebook_path", "target_path"]
        .iter()
        .find_map(|k| get(&p.input, k))
        .map(js_string);

    if let Some(allowed_tools) = &p.allowed_tools {
        let mut allowed: Vec<String> = allowed_tools
            .iter()
            .map(|t| role_tool_category(t))
            .collect();
        if category == "verify" && allowed.iter().any(|a| a == "shell") {
            allowed.push("verify".into());
        }
        let exact = format!("exact:{}", name.to_ascii_lowercase());
        if !allowed.contains(&category) && !allowed.contains(&exact) {
            let shown = if name.is_empty() {
                "unknown tool"
            } else {
                name
            };
            return deny(format!(
                "Denied by Field: {shown} is not in this role's tool capability set."
            ));
        }
    }

    if category == "web_fetch" {
        if let Some(problem) = validate_egress_url(get(&p.input, "url"), &p.allowed_domains) {
            return deny(format!("Denied by Field: {problem}"));
        }
    }

    let command = || {
        get(&p.input, "command")
            .or(get(&p.input, "cmd"))
            .map(js_string)
            .unwrap_or_default()
    };
    if category == "shell" && network_shell().is_match(&command()) {
        return deny("Denied by Field: network-capable shell commands cannot enforce destination policy; use WebFetch with a declared domain.");
    }

    if let (Some(candidate), Some(root)) = (&candidate, p.workspace_path) {
        let resolved = resolve_under(root, candidate);
        if !inside(root, &resolved) {
            return deny("Denied by Field: target escapes the assigned workspace.");
        }
        if matches!(category.as_str(), "edit" | "write") {
            if let Some(validate) = p.validate_write_path {
                if validate(&relative_to(root, &resolved)).is_err() {
                    return deny(
                        "Denied by Field: target violates workspace path or sensitive-file policy.",
                    );
                }
            }
        }
    }

    if mutating
        && matches!(
            p.environment_scope,
            Some("snapshot" | "production-readonly")
        )
    {
        return deny(format!(
            "Denied by Field: {} campaigns are read-only.",
            p.environment_scope.unwrap_or("")
        ));
    }

    if matches!(category.as_str(), "edit" | "write") {
        if let Some(scope) = p.write_scope.as_ref().filter(|s| !s.is_empty()) {
            let (Some(candidate), Some(root)) = (&candidate, p.workspace_path) else {
                return deny("Denied by Field: the scoped write has no verifiable workspace path.");
            };
            let relative = relative_to(root, &resolve_under(root, candidate));
            let matcher = build_matcher(scope);
            if relative.is_empty()
                || relative == ".."
                || relative.starts_with("../")
                || !matcher(&relative)
            {
                return deny(format!(
                    "Denied by Field: target is outside this role's write scope ({}).",
                    scope.join(", ")
                ));
            }
        }
    }

    if p.read_only && matches!(category.as_str(), "edit" | "write") {
        return deny("Denied by Field: this campaign role is read-only.");
    }

    if p.read_only && category == "shell" {
        let command = command();
        if command.is_empty()
            || mutating_shell().is_match(&command)
            || escapes_scope().is_match(&command)
        {
            return deny("Denied by Field: read-only shell requests may inspect or test, but may not mutate or escape scope.");
        }
    }
    None
}

/// Why an outbound URL is refused, or `None` when it is allowed.
pub fn validate_egress_url(value: Option<&Value>, allowed_domains: &[String]) -> Option<String> {
    let text = value.map(js_string).unwrap_or_default();
    let Ok(url) = url::Url::parse(&text) else {
        return Some("network destination must be a complete HTTP(S) URL.".into());
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Some("network destination must be credential-free HTTP(S).".into());
    }
    let hostname = url
        .host_str()
        .unwrap_or("")
        .to_ascii_lowercase()
        .trim_end_matches('.')
        .to_string();
    if hostname.is_empty() || is_private_host(&hostname) {
        return Some(
            "loopback, private, link-local, and metadata destinations are blocked.".into(),
        );
    }
    let declared: Vec<String> = allowed_domains
        .iter()
        .map(|d| {
            d.to_ascii_lowercase()
                .trim_start_matches("*.")
                .trim_end_matches('.')
                .to_string()
        })
        .collect();
    if !declared
        .iter()
        .any(|d| hostname == *d || hostname.ends_with(&format!(".{d}")))
    {
        return Some(format!(
            "destination {hostname} is not declared in this role's network policy."
        ));
    }
    None
}

fn is_private_host(hostname: &str) -> bool {
    if hostname == "localhost"
        || hostname.ends_with(".localhost")
        || hostname == "metadata.google.internal"
    {
        return true;
    }
    let bare = hostname.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => {
            let [a, b, _, _] = v4.octets();
            a == 0
                || a == 10
                || a == 127
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 168)
                || a >= 224
        }
        Ok(IpAddr::V6(v6)) => {
            let text = v6.to_string();
            v6.is_unspecified()
                || v6.is_loopback()
                || text.starts_with("fc")
                || text.starts_with("fd")
                || text.starts_with("fe8")
                || text.starts_with("fe9")
                || text.starts_with("fea")
                || text.starts_with("feb")
                || text.starts_with("::ffff:")
                || v6.to_ipv4_mapped().is_some()
        }
        Err(_) => false,
    }
}

/// Parse one structured campaign report from a final harness message. The
/// surrounding prose is ignored; the sentinel keeps ordinary JSON examples
/// from being read as operational commands.
pub fn parse_campaign_report(value: Option<&Value>) -> Option<Value> {
    let text = value.map(js_string).unwrap_or_default();
    let marker = text.rfind("FIELD_REPORT:")?;
    let mut tail = text[marker + "FIELD_REPORT:".len()..].trim().to_string();
    static FENCED: OnceLock<Regex> = OnceLock::new();
    let fenced = FENCED
        .get_or_init(|| Regex::new(r"(?is)^```(?:json)?\s*(.*?)\s*```").expect("static regex"));
    if let Some(caps) = fenced.captures(&tail) {
        tail = caps[1].to_string();
    } else {
        tail = tail.lines().next().unwrap_or("").to_string();
    }
    match serde_json::from_str::<Value>(&tail) {
        Ok(v) if v.is_object() => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    //! `field/server/test/permission-policy.test.mjs`, case for case.
    use super::*;
    use serde_json::json;

    fn root() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\workspace\project")
        } else {
            PathBuf::from("/workspace/project")
        }
    }

    fn check(tool: &str, input: Value, tweak: impl FnOnce(&mut PolicyInput<'_>)) -> Option<Denial> {
        let r = root();
        let mut p = PolicyInput {
            tool_name: tool,
            input,
            workspace_path: Some(&r),
            read_only: false,
            environment_scope: Some("sandbox"),
            ..Default::default()
        };
        tweak(&mut p);
        evaluate_permission_policy(&p)
    }

    fn msg(d: Option<Denial>) -> String {
        d.expect("a denial").message
    }

    #[test]
    fn escapes_scopes_and_read_only_roles() {
        assert!(msg(check(
            "Write",
            json!({"file_path": "../outside.txt"}),
            |_| {}
        ))
        .contains("escapes"));
        assert!(msg(check(
            "Write",
            json!({"file_path": "src/inside.txt"}),
            |p| p.environment_scope = Some("production-readonly")
        ))
        .contains("read-only"));
        assert!(
            msg(check("Edit", json!({"file_path": "src/inside.txt"}), |p| {
                p.read_only = true
            }))
            .contains("role is read-only")
        );
        assert!(msg(check("Bash", json!({"command": "rm -rf build"}), |p| p
            .read_only =
            true))
        .contains("may not mutate"));
        assert!(msg(check(
            "Bash",
            json!({"command": "Get-Content ..\\secrets.env"}),
            |p| p.read_only = true
        ))
        .contains("escape scope"));
        assert!(
            check("Bash", json!({"command": "npm test"}), |p| p.read_only =
                true)
            .is_none()
        );
        assert!(check("Write", json!({"file_path": "src/inside.txt"}), |_| {}).is_none());
        assert!(
            msg(check("Read", json!({"file_path": "../outside.txt"}), |p| {
                p.read_only = true
            }))
            .contains("escapes")
        );
    }

    #[test]
    fn capability_sets_and_write_scopes() {
        let ro = || Some(vec!["Read".to_string(), "Grep".into(), "Glob".into()]);
        assert!(
            msg(check("write_file", json!({"path": "src/code.rs"}), |p| {
                p.allowed_tools = ro();
                p.read_only = true;
            }))
            .contains("not in this role's tool capability set")
        );
        assert!(msg(check(
            "WebFetch",
            json!({"url": "https://example.com"}),
            |p| p.allowed_tools = ro()
        ))
        .contains("not in this role's tool capability set"));
        assert!(
            check("search", json!({"query": "needle"}), |p| {
                p.allowed_tools = ro();
                p.read_only = true;
            })
            .is_none(),
            "Knossos search maps to the declared Grep capability"
        );
        let rw = || Some(vec!["Read".to_string(), "Write".into()]);
        assert!(check("write_file", json!({"path": "docs/plan.md"}), |p| {
            p.allowed_tools = rw();
            p.write_scope = Some(vec!["**/*.md".into()]);
        })
        .is_none());
        assert!(
            msg(check("write_file", json!({"path": "src/code.rs"}), |p| {
                p.allowed_tools = rw();
                p.write_scope = Some(vec!["**/*.md".into()]);
            }))
            .contains("outside this role's write scope")
        );
        assert!(msg(check("write_file", json!({}), |p| {
            p.allowed_tools = rw();
            p.write_scope = Some(vec!["**/*.md".into()]);
        }))
        .contains("no verifiable workspace path"));
        fn junction(_: &str) -> Result<(), String> {
            Err("junction".into())
        }
        assert!(
            msg(check("write_file", json!({"path": "docs/plan.md"}), |p| {
                p.allowed_tools = rw();
                p.validate_write_path = Some(&junction);
            }))
            .contains("workspace path or sensitive-file policy")
        );
    }

    #[test]
    fn egress_policy() {
        let d = |s: &str| vec![s.to_string()];
        assert_eq!(
            validate_egress_url(
                Some(&json!("https://docs.claude.com/en/docs")),
                &d("docs.claude.com")
            ),
            None
        );
        assert_eq!(
            validate_egress_url(
                Some(&json!("https://sub.docs.claude.com/page")),
                &d("docs.claude.com")
            ),
            None
        );
        assert!(
            validate_egress_url(Some(&json!("http://127.0.0.1:8080/admin")), &d("127.0.0.1"))
                .unwrap()
                .contains("blocked")
        );
        assert!(
            validate_egress_url(Some(&json!("http://[::1]:8080/admin")), &d("::1"))
                .unwrap()
                .contains("blocked")
        );
        assert!(validate_egress_url(
            Some(&json!("http://169.254.169.254/latest/meta-data")),
            &d("169.254.169.254")
        )
        .unwrap()
        .contains("blocked"));
        assert!(validate_egress_url(
            Some(&json!("https://user:pass@docs.claude.com/")),
            &d("docs.claude.com")
        )
        .unwrap()
        .contains("credential-free"));
        assert!(
            validate_egress_url(Some(&json!("file:///etc/passwd")), &d("docs.claude.com"))
                .unwrap()
                .contains("credential-free")
        );
        assert!(
            validate_egress_url(Some(&json!("https://evil.example/")), &d("docs.claude.com"))
                .unwrap()
                .contains("not declared")
        );
        assert!(check(
            "WebFetch",
            json!({"url": "https://docs.claude.com/en/docs"}),
            |p| {
                p.allowed_tools = Some(vec!["WebFetch".into()]);
                p.allowed_domains = d("docs.claude.com");
            }
        )
        .is_none());
        assert!(msg(check(
            "WebFetch",
            json!({"url": "http://localhost:8080/secrets"}),
            |p| {
                p.allowed_tools = Some(vec!["WebFetch".into()]);
                p.allowed_domains = d("localhost");
            }
        ))
        .contains("blocked"));
        assert!(msg(check(
            "Bash",
            json!({"command": "curl https://docs.claude.com"}),
            |p| {
                p.allowed_tools = Some(vec!["Bash".into()]);
                p.allowed_domains = d("docs.claude.com");
            }
        ))
        .contains("network-capable shell"));
    }

    #[test]
    fn deny_lists_narrow_the_role_and_always_remove_unmanaged_delegation() {
        assert_eq!(
            deny_list_for(
                Some(&json!({ "tools_allow": ["Read", "Grep", "Edit", "Bash"] })),
                Some(&json!({ "tools_allow": ["Read", "Grep"] })),
            ),
            Some(vec![
                "Edit".into(),
                "Bash".into(),
                "Task".into(),
                "Agent".into()
            ])
        );
        assert_eq!(
            deny_list_for(
                Some(&json!({ "read_only": true, "tools_allow": ["Read", "Grep", "WebSearch"] })),
                Some(&json!({ "tools_allow": ["Read"] })),
            ),
            Some(vec![
                "Edit".into(),
                "Write".into(),
                "NotebookEdit".into(),
                "PowerShell".into(),
                "Grep".into(),
                "WebSearch".into(),
                "Task".into(),
                "Agent".into()
            ])
        );
    }

    #[test]
    fn campaign_reports_need_the_sentinel() {
        assert!(parse_campaign_report(Some(&json!("all done {\"kind\":\"x\"}"))).is_none());
        let report = parse_campaign_report(Some(&json!(
            "complete\nFIELD_REPORT: {\"kind\":\"objective_satisfied\",\"evidence\":[\"green\"]}"
        )))
        .unwrap();
        assert_eq!(report["kind"], "objective_satisfied");
        let fenced = parse_campaign_report(Some(&json!(
            "FIELD_REPORT:\n```json\n{\"kind\": \"finding\"}\n```"
        )))
        .unwrap();
        assert_eq!(fenced["kind"], "finding");
    }

    #[test]
    fn globs_compile_in_one_pass() {
        let m = build_matcher(&["**/node_modules/**".to_string(), "*.md".to_string()]);
        assert!(m("a/b/node_modules/x.js"));
        assert!(m("node_modules"), "the directory node itself is matched");
        assert!(m("README.md"));
        assert!(!m("docs/README.md"));
    }
}
