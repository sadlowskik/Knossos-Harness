//! Workspace paths: what the browser may see and touch inside a mounted
//! workspace. Port of `field/server/src/workspace-path.js` (its glob matcher
//! already lives in `policy.rs`), plus the `/api/fs/*` route bodies from
//! `field/server/src/api.js`.
//!
//! Every request path is normalised, checked against the workspace security
//! policy (default secrets, the operator's deny globs, every `.gitignore` on
//! the way down), walked component by component so no symlink or junction
//! can lead out of the root, and finally canonicalised and checked again.

use super::config::{dunce_canonicalize, FieldSettings};
use super::js::{get, get_arr, get_bool, get_str, js_string};
use super::policy::build_matcher;
use regex::Regex;
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

/// Directory names a tree listing never descends into.
pub const SKIP_DIRS: [&str; 5] = [".git", "node_modules", "target", "dist", ".field-state"];
/// Files above this size are reported as `tooLarge` rather than read.
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

const PRIVATE_BASENAMES: [&str; 19] = [
    ".git",
    ".field-state",
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".npmrc",
    ".yarnrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    "credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
];
const PRIVATE_EXTENSIONS: [&str; 7] =
    [".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".kdbx"];
const MAX_IGNORE_FILE_BYTES: u64 = 1024 * 1024;

/// A workspace path error: the message the Node server threw, which the API
/// reports as `400 bad_request`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct WorkspaceError(pub String);

fn err<T>(message: impl Into<String>) -> Result<T, WorkspaceError> {
    Err(WorkspaceError(message.into()))
}

/// Node's `ENOENT: no such file or directory, lstat '/path'` shape, as far
/// as `std::io` lets us reproduce it.
fn io_message(op: &str, path: &Path, error: &std::io::Error) -> WorkspaceError {
    use std::io::ErrorKind;
    let code = match error.kind() {
        ErrorKind::NotFound => "ENOENT: no such file or directory".to_string(),
        ErrorKind::PermissionDenied => "EACCES: permission denied".to_string(),
        ErrorKind::AlreadyExists => "EEXIST: file already exists".to_string(),
        other => format!("{other:?}: {error}"),
    };
    WorkspaceError(format!("{code}, {op} '{}'", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Operation {
    #[default]
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetType {
    #[default]
    Any,
    File,
    Directory,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResolveOptions {
    pub operation: Operation,
    pub target: TargetType,
    pub allow_sensitive: bool,
}

impl ResolveOptions {
    pub fn file() -> Self {
        ResolveOptions {
            target: TargetType::File,
            ..Default::default()
        }
    }

    pub fn directory() -> Self {
        ResolveOptions {
            target: TargetType::Directory,
            ..Default::default()
        }
    }

    pub fn write_file() -> Self {
        ResolveOptions {
            operation: Operation::Write,
            target: TargetType::File,
            allow_sensitive: false,
        }
    }
}

/// A request path resolved inside a mounted workspace.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// The workspace record from `field.yaml`, canonicalised.
    pub ws: Value,
    pub root: PathBuf,
    /// Canonical when the target exists; the intended location otherwise.
    pub abs: PathBuf,
    /// Workspace-relative, `/`-separated.
    pub relative: String,
    pub exists: bool,
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
}

// ---- path keys and normalisation -----------------------------------------

fn path_key(value: &Path) -> String {
    let text = value.to_string_lossy().into_owned();
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

fn is_within(root: &Path, candidate: &Path) -> bool {
    let root_key = path_key(root);
    let candidate_key = path_key(candidate);
    candidate_key == root_key
        || candidate_key.starts_with(&format!("{root_key}{}", std::path::MAIN_SEPARATOR))
}

/// `path.normalize` on a request path, as components. Rejects absolute and
/// NUL-bearing input and anything that climbs above the root.
fn normalize_relative(value: Option<&str>) -> Result<Vec<String>, WorkspaceError> {
    let Some(value) = value.filter(|v| !v.is_empty()) else {
        return Ok(Vec::new());
    };
    let path = Path::new(value);
    let prefixed = matches!(path.components().next(), Some(Component::Prefix(_)));
    if value.contains('\0') || path.is_absolute() || path.has_root() || prefixed {
        return err("workspace paths must be relative");
    }
    let mut out: Vec<String> = Vec::new();
    for part in value.split(std::path::is_separator) {
        match part {
            "" | "." => {}
            ".." => match out.last() {
                Some(last) if last != ".." => {
                    out.pop();
                }
                _ => out.push("..".into()),
            },
            other => out.push(other.into()),
        }
    }
    if out.first().map(String::as_str) == Some("..") {
        return err("path escapes the workspace root");
    }
    Ok(out)
}

fn extname(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) | None => "",
        Some(i) => &name[i..],
    }
}

fn sensitive_default(relative: &str) -> bool {
    let normalized = relative.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').filter(|p| !p.is_empty()).collect();
    for (index, part) in parts.iter().enumerate() {
        let name = part.to_lowercase();
        if PRIVATE_BASENAMES.contains(&name.as_str()) {
            return true;
        }
        if name == ".config"
            && parts
                .get(index + 1)
                .map(|n| n.to_lowercase() == "gcloud")
                .unwrap_or(false)
        {
            return true;
        }
        if name.starts_with(".env") {
            return true;
        }
        if PRIVATE_EXTENSIONS.contains(&extname(&name)) {
            return true;
        }
    }
    false
}

// ---- .gitignore rules ------------------------------------------------------

/// One compiled `.gitignore` line, with the `ignore` package's semantics: a
/// pattern without a slash matches at any depth, a trailing slash matches
/// directories only, and a match covers everything below it.
#[derive(Debug)]
struct IgnoreRule {
    negative: bool,
    regex: Regex,
}

#[derive(Debug, Default)]
struct IgnoreRules {
    rules: Vec<IgnoreRule>,
}

#[derive(Debug, Clone, Copy, Default)]
struct IgnoreResult {
    ignored: bool,
    unignored: bool,
}

fn gitignore_body_regex(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' if i + 1 < chars.len() => {
                out.push_str(&regex::escape(&chars[i + 1].to_string()));
                i += 2;
            }
            '*' => {
                if chars.get(i + 1) == Some(&'*') {
                    if chars.get(i + 2) == Some(&'/') {
                        out.push_str("(?:.*/)?");
                        i += 3;
                    } else {
                        out.push_str(".*");
                        i += 2;
                    }
                } else {
                    out.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                out.push_str("[^/]");
                i += 1;
            }
            '[' => {
                let close = chars[i + 1..].iter().position(|c| *c == ']');
                match close {
                    Some(offset) if offset > 0 => {
                        let mut class = String::from("[");
                        for (j, cc) in chars[i + 1..i + 1 + offset].iter().enumerate() {
                            match cc {
                                '!' if j == 0 => class.push('^'),
                                '\\' | '[' => {
                                    class.push('\\');
                                    class.push(*cc);
                                }
                                other => class.push(*other),
                            }
                        }
                        class.push(']');
                        out.push_str(&class);
                        i += offset + 2;
                    }
                    _ => {
                        out.push_str("\\[");
                        i += 1;
                    }
                }
            }
            other => {
                out.push_str(&regex::escape(&other.to_string()));
                i += 1;
            }
        }
    }
    out
}

fn compile_ignore_line(line: &str) -> Option<IgnoreRule> {
    let mut pattern = line.trim_end_matches(['\r', '\n']).to_string();
    while pattern.ends_with(' ') && !pattern.ends_with("\\ ") {
        pattern.pop();
    }
    if pattern.is_empty() || pattern.starts_with('#') {
        return None;
    }
    let negative = pattern.starts_with('!');
    if negative {
        pattern.remove(0);
    }
    let dir_only = pattern.ends_with('/');
    let body = pattern.trim_end_matches('/');
    let anchored = body.contains('/');
    let body = body.trim_start_matches('/');
    if body.is_empty() {
        return None;
    }
    let prefix = if anchored { "^" } else { "^(?:.*/)?" };
    let suffix = if dir_only { "/.*$" } else { "(?:/.*)?$" };
    let source = format!("{prefix}{}{suffix}", gitignore_body_regex(body));
    let regex = Regex::new(&source).ok()?;
    Some(IgnoreRule { negative, regex })
}

impl IgnoreRules {
    fn parse(text: &str) -> Self {
        IgnoreRules {
            rules: text.lines().filter_map(compile_ignore_line).collect(),
        }
    }

    fn test_one(&self, path: &str) -> IgnoreResult {
        let mut result = IgnoreResult::default();
        for rule in &self.rules {
            if rule.regex.is_match(path) {
                result.ignored = !rule.negative;
                result.unignored = rule.negative;
            }
        }
        result
    }

    /// An excluded parent excludes everything below it, whatever later
    /// negations say.
    fn test(&self, path: &str) -> IgnoreResult {
        let trimmed = path.trim_end_matches('/');
        let parts: Vec<&str> = trimmed.split('/').collect();
        for depth in 1..parts.len() {
            let parent = format!("{}/", parts[..depth].join("/"));
            let result = self.test_one(&parent);
            if result.ignored {
                return result;
            }
        }
        self.test_one(path)
    }
}

/// The `.gitignore` in `dir`: empty when there is none, `None` when it is
/// unreadable, a link, or implausibly large (the path is then hidden).
fn gitignore_rules(dir: &Path) -> Option<IgnoreRules> {
    let file = dir.join(".gitignore");
    let link_meta = match fs::symlink_metadata(&file) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(IgnoreRules::default()),
        Err(_) => return None,
    };
    if !link_meta.is_file() {
        return None;
    }
    let mut handle = open_nofollow_read(&file).ok()?;
    let meta = handle.metadata().ok()?;
    if !meta.is_file() || meta.len() > MAX_IGNORE_FILE_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    handle.read_to_end(&mut bytes).ok()?;
    Some(IgnoreRules::parse(&String::from_utf8_lossy(&bytes)))
}

/// `ignore.isPathValid`: relative, non-empty, not `.`/`..`, not `./x`.
fn is_ignore_path_valid(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    let dots = path.chars().take_while(|c| *c == '.').count();
    if dots == path.len() {
        return false;
    }
    path.chars().nth(dots) != Some('/')
}

// ---- visibility ------------------------------------------------------------

/// The security policy of one workspace: `ws.isVisible` in the Node server.
pub struct Visibility {
    root: Option<PathBuf>,
    custom: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl std::fmt::Debug for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Visibility")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl Visibility {
    /// The policy for a workspace record, with the operator's global
    /// `field.sensitive_names` and the workspace's own `sensitive.deny`.
    pub fn for_workspace(settings: &FieldSettings, ws: &Value) -> Visibility {
        let mut deny: Vec<String> = get_arr(&settings.field, "sensitive_names")
            .map(|list| list.iter().map(js_string).collect())
            .unwrap_or_default();
        if let Some(list) = get(ws, "sensitive").and_then(|s| get_arr(s, "deny")) {
            deny.extend(list.iter().map(js_string));
        }
        let mounted = get_bool(ws, "mounted") == Some(true);
        let root = get_str(ws, "canonicalPath")
            .filter(|_| mounted)
            .map(PathBuf::from);
        Visibility::new(root, &deny)
    }

    /// A policy over a root directory (`None` for an unmounted workspace,
    /// which hides everything).
    pub fn new(root: Option<PathBuf>, custom_deny: &[String]) -> Visibility {
        let patterns: Vec<String> = custom_deny.iter().map(|p| p.replace('\\', "/")).collect();
        Visibility {
            root,
            custom: Box::new(build_matcher(&patterns)),
        }
    }

    pub fn is_visible(&self, relative: &str) -> bool {
        let Some(root) = &self.root else {
            return false;
        };
        let normalized = relative.replace('\\', "/");
        let normalized = normalized.strip_prefix("./").unwrap_or(&normalized);
        if normalized.is_empty()
            || !is_ignore_path_valid(normalized)
            || sensitive_default(normalized)
            || (self.custom)(normalized)
        {
            return false;
        }
        let parts: Vec<&str> = normalized.trim_end_matches('/').split('/').collect();
        let mut policies: Vec<(usize, IgnoreRules)> = Vec::new();
        // Parent exclusions cannot be undone by descendant negations. Reload
        // policy so a newly ignored credential is hidden without a restart.
        for depth in 0..parts.len() {
            let dir = parts[..depth].iter().fold(root.clone(), |p, s| p.join(s));
            let Some(rules) = gitignore_rules(&dir) else {
                return false;
            };
            policies.push((depth, rules));
            let target = dir.join(parts[depth]);
            let stat = match fs::symlink_metadata(&target) {
                Ok(meta) => Some(meta),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(_) => return false,
            };
            if stat
                .as_ref()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                return false;
            }
            let directory = depth < parts.len() - 1
                || stat.as_ref().map(|m| m.is_dir()).unwrap_or(false)
                || normalized.ends_with('/');
            let mut ignored = false;
            for (policy_depth, rules) in &policies {
                let mut candidate = parts[*policy_depth..=depth].join("/");
                if directory {
                    candidate.push('/');
                }
                let result = rules.test(&candidate);
                if result.ignored {
                    ignored = true;
                } else if result.unignored {
                    ignored = false;
                }
            }
            if ignored {
                return false;
            }
        }
        true
    }
}

// ---- resolution ------------------------------------------------------------

struct Chain {
    missing_leaf: bool,
}

fn inspect_chain(
    root: &Path,
    parts: &[String],
    allow_missing_leaf: bool,
) -> Result<Chain, WorkspaceError> {
    let mut cursor = root.to_path_buf();
    for (index, part) in parts.iter().enumerate() {
        cursor.push(part);
        let leaf = index == parts.len() - 1;
        let stat = match fs::symlink_metadata(&cursor) {
            Ok(meta) => meta,
            Err(e) => {
                if allow_missing_leaf && leaf && e.kind() == std::io::ErrorKind::NotFound {
                    return Ok(Chain { missing_leaf: true });
                }
                return Err(io_message("lstat", &cursor, &e));
            }
        };
        if stat.file_type().is_symlink() {
            return err("symlink or junction traversal is not allowed");
        }
        if !leaf && !stat.is_dir() {
            return err("workspace path has a non-directory parent");
        }
    }
    Ok(Chain {
        missing_leaf: false,
    })
}

fn shown(value: Option<&str>) -> &str {
    value.unwrap_or("null")
}

/// `resolveWorkspacePath`: the one gate every workspace file access passes.
pub fn resolve_workspace_path(
    settings: &FieldSettings,
    workspace_id: Option<&str>,
    relative: Option<&str>,
    options: ResolveOptions,
) -> Result<Resolved, WorkspaceError> {
    let Some(ws) = workspace_id.and_then(|id| settings.workspace(id)) else {
        return err(format!("unknown workspace: {}", shown(workspace_id)));
    };
    let ws_id = shown(workspace_id).to_string();
    let canonical_root = get_str(ws, "canonicalPath");
    let Some(root) = canonical_root.filter(|_| get_bool(ws, "mounted") == Some(true)) else {
        return err(format!("workspace {ws_id} is not mounted"));
    };
    let parts = normalize_relative(relative)?;
    let display = parts.join("/");
    if !options.allow_sensitive
        && !display.is_empty()
        && !Visibility::for_workspace(settings, ws).is_visible(&display)
    {
        return err("path is hidden by the workspace security policy");
    }

    let root = PathBuf::from(root);
    let absolute = parts.iter().fold(root.clone(), |p, s| p.join(s));
    if !is_within(&root, &absolute) {
        return err("path escapes the workspace root");
    }
    let chain = inspect_chain(&root, &parts, options.operation == Operation::Write)?;
    if chain.missing_leaf {
        if options.target == TargetType::Directory {
            return err("workspace directory does not exist");
        }
        return Ok(Resolved {
            ws: ws.clone(),
            root,
            abs: absolute,
            relative: display,
            exists: false,
            size: 0,
            is_file: false,
            is_dir: false,
        });
    }

    let canonical = dunce_canonicalize(&absolute).ok_or_else(|| {
        WorkspaceError(format!(
            "ENOENT: no such file or directory, realpath '{}'",
            absolute.display()
        ))
    })?;
    if !is_within(&root, &canonical) {
        return err("canonical path escapes the workspace root");
    }
    let stat = fs::metadata(&canonical).map_err(|e| io_message("stat", &canonical, &e))?;
    let (is_file, is_dir) = (stat.is_file(), stat.is_dir());
    match options.target {
        TargetType::File if !is_file => return err("workspace target is not a regular file"),
        TargetType::Directory if !is_dir => return err("workspace target is not a directory"),
        TargetType::Any if !is_file && !is_dir => return err("unsupported workspace file type"),
        _ => {}
    }
    Ok(Resolved {
        ws: ws.clone(),
        root,
        abs: canonical,
        relative: display,
        exists: true,
        size: stat.len(),
        is_file,
        is_dir,
    })
}

// ---- file access -----------------------------------------------------------

/// `O_NOFOLLOW` where the platform has it; Windows opens the reparse point
/// itself, which the `is_file` check after the open then rejects.
fn nofollow(options: &mut fs::OpenOptions) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0o400000);
    }
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x0100);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = options;
    }
}

fn open_nofollow_read(path: &Path) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    nofollow(&mut options);
    options.open(path)
}

/// `readWorkspaceFile`: read-only, never through a link, UTF-8 (lossy).
pub fn read_workspace_file(resolved: &Resolved) -> Result<String, WorkspaceError> {
    let mut handle =
        open_nofollow_read(&resolved.abs).map_err(|e| io_message("open", &resolved.abs, &e))?;
    let meta = handle
        .metadata()
        .map_err(|e| io_message("fstat", &resolved.abs, &e))?;
    if !meta.is_file() {
        return err("workspace target is not a regular file");
    }
    let mut bytes = Vec::new();
    handle
        .read_to_end(&mut bytes)
        .map_err(|e| io_message("read", &resolved.abs, &e))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// `writeWorkspaceFile`: create or truncate, mode 0600, never through a link.
pub fn write_workspace_file(resolved: &Resolved, content: &str) -> Result<(), WorkspaceError> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    nofollow(&mut options);
    let mut handle = options
        .open(&resolved.abs)
        .map_err(|e| io_message("open", &resolved.abs, &e))?;
    let meta = handle
        .metadata()
        .map_err(|e| io_message("fstat", &resolved.abs, &e))?;
    if !meta.is_file() {
        return err("workspace target is not a regular file");
    }
    handle
        .write_all(content.as_bytes())
        .map_err(|e| io_message("write", &resolved.abs, &e))
}

// ---- the /api/fs routes ----------------------------------------------------

/// `path.posix.join(a, b)`: joined and normalised, `/`-separated.
fn posix_join(base: &str, name: &str) -> String {
    let joined = format!("{base}/{name}");
    let mut out: Vec<&str> = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => match out.last() {
                Some(last) if *last != ".." => {
                    out.pop();
                }
                _ => out.push(".."),
            },
            other => out.push(other),
        }
    }
    if out.is_empty() {
        ".".into()
    } else {
        out.join("/")
    }
}

/// `localeCompare` for file names, near enough: case-insensitive first,
/// then byte order as the tie-break.
fn locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    a.to_lowercase()
        .cmp(&b.to_lowercase())
        .then_with(|| a.cmp(b))
}

/// `GET /api/fs/tree`.
pub fn fs_tree(
    settings: &FieldSettings,
    ws_id: Option<&str>,
    dir: Option<&str>,
) -> Result<Value, WorkspaceError> {
    let rel = dir.unwrap_or("");
    let resolved = resolve_workspace_path(settings, ws_id, Some(rel), ResolveOptions::directory())?;
    let visibility = Visibility::for_workspace(settings, &resolved.ws);
    let rel_posix = rel.replace('\\', "/");
    let read_dir =
        fs::read_dir(&resolved.abs).map_err(|e| io_message("scandir", &resolved.abs, &e))?;
    let mut entries: Vec<(String, bool, Value)> = Vec::new();
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        let child_rel = posix_join(&rel_posix, &name);
        if !visibility.is_visible(&child_rel) {
            continue;
        }
        let file_type = entry.file_type().ok();
        let is_file = file_type.map(|t| t.is_file()).unwrap_or(false);
        let is_dir = file_type.map(|t| t.is_dir()).unwrap_or(false);
        let size = if is_file {
            fs::metadata(resolved.abs.join(&name)).ok().map(|m| m.len())
        } else {
            None
        };
        entries.push((
            name.clone(),
            is_dir,
            json!({ "name": name, "path": child_rel, "dir": is_dir, "size": size }),
        ));
    }
    entries.sort_by(|a, b| {
        if a.1 == b.1 {
            locale_compare(&a.0, &b.0)
        } else if a.1 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });
    Ok(json!({
        "ws": ws_id,
        "dir": rel,
        "entries": entries.into_iter().map(|(_, _, v)| v).collect::<Vec<_>>(),
    }))
}

/// `GET /api/fs/file`.
pub fn fs_file(
    settings: &FieldSettings,
    ws_id: Option<&str>,
    rel: Option<&str>,
) -> Result<Value, WorkspaceError> {
    let resolved = resolve_workspace_path(settings, ws_id, rel, ResolveOptions::file())?;
    if resolved.size > MAX_FILE_BYTES {
        return Ok(json!({
            "ws": ws_id, "path": rel, "tooLarge": true, "size": resolved.size, "content": Value::Null,
        }));
    }
    let content = read_workspace_file(&resolved)?;
    Ok(json!({
        "ws": ws_id, "path": rel, "size": resolved.size, "tooLarge": false, "content": content,
    }))
}

/// `PUT /api/fs/file`. `body.content` defaults to the empty string.
pub fn fs_put_file(settings: &FieldSettings, body: &Value) -> Result<Value, WorkspaceError> {
    let content = match get(body, "content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => return err("The \"data\" argument must be of type string"),
    };
    let resolved = resolve_workspace_path(
        settings,
        get_str(body, "ws"),
        get_str(body, "path"),
        ResolveOptions::write_file(),
    )?;
    write_workspace_file(&resolved, &content)?;
    // The fs watcher will observe this write and emit the change event itself.
    Ok(json!({ "ok": true, "bytes": content.len() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::config::canonicalize_workspace;

    fn settings_for(root: &Path, field_dir: &Path) -> FieldSettings {
        let ws = canonicalize_workspace(
            &json!({ "id": "fixture", "path": root.to_string_lossy() }),
            field_dir,
        );
        FieldSettings {
            field_dir: field_dir.to_path_buf(),
            field: json!({}),
            defaults: json!({}),
            workspaces: vec![ws],
            endpoints: vec![],
            websites: vec![],
            roles: vec![],
            agents: vec![],
            missions: vec![],
            constitutions: vec![],
            skills: vec![],
            routines: vec![],
            memory: vec![],
        }
    }

    fn write(path: &Path, text: &str) {
        fs::write(path, text).unwrap();
    }

    /// `workspace-path.test.mjs`: canonical roots, secrets, gitignore, writes,
    /// and link containment.
    #[test]
    fn workspace_paths_match_the_node_oracle() {
        let fixture = tempfile::tempdir().unwrap();
        let root = fixture.path().join("workspace");
        let outside = fixture.path().join("outside");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        write(&root.join("src").join("visible.txt"), "visible");
        write(&root.join(".env.local"), "SECRET=canary");
        write(&root.join("private.pem"), "private");
        write(&root.join("ignored.txt"), "ignored");
        write(
            &root.join(".gitignore"),
            "ignored.txt\nbuild/\n!build/keep.txt\n",
        );
        fs::create_dir_all(root.join("build")).unwrap();
        write(&root.join("build").join("drop.txt"), "drop");
        write(&root.join("build").join("keep.txt"), "keep");
        write(&outside.join("escape.txt"), "escape");

        let settings = settings_for(&root, fixture.path());
        let ws = &settings.workspaces[0];
        assert_eq!(get_bool(ws, "mounted"), Some(true));
        assert_eq!(
            get_str(ws, "path").map(PathBuf::from),
            dunce_canonicalize(&root)
        );
        let visibility = || Visibility::for_workspace(&settings, ws);

        let visible = resolve_workspace_path(
            &settings,
            Some("fixture"),
            Some("src/visible.txt"),
            ResolveOptions::file(),
        )
        .unwrap();
        assert_eq!(read_workspace_file(&visible).unwrap(), "visible");
        assert!(visibility().is_visible("src/visible.txt"));
        assert!(!visibility().is_visible("ignored.txt"));
        assert!(!visibility().is_visible("build/drop.txt"));
        assert!(!visibility().is_visible(".env.local"));
        assert!(!visibility().is_visible("private.pem"));
        assert!(!visibility().is_visible(".envrc"));
        assert!(
            !visibility().is_visible("build/keep.txt"),
            "excluded parent cannot be resurrected"
        );
        write(
            &root.join("src").join(".gitignore"),
            "/local.txt\n*.secret\n!keep.secret\n",
        );
        assert!(!visibility().is_visible("src/local.txt"));
        assert!(
            visibility().is_visible("local.txt"),
            "nested policy stays scoped"
        );
        assert!(!visibility().is_visible("src/hidden.secret"));
        assert!(visibility().is_visible("src/keep.secret"));
        fs::create_dir_all(root.join("src").join("nested")).unwrap();
        assert!(
            visibility().is_visible("src/nested/local.txt"),
            "anchored rule is directory relative"
        );
        write(
            &root.join("src").join("nested").join(".gitignore"),
            "!keep.secret\n",
        );
        assert!(visibility().is_visible("src/nested/keep.secret"));
        write(
            &root.join("src").join(".gitignore"),
            "/local.txt\n*.secret\n!keep.secret\nvisible.txt\n",
        );
        assert!(
            !visibility().is_visible("src/visible.txt"),
            "policy edits apply immediately"
        );
        write(&root.join("src").join(".gitignore"), "");
        assert!(visibility().is_visible("src/visible.txt"));

        let escape = resolve_workspace_path(
            &settings,
            Some("fixture"),
            Some("../outside/escape.txt"),
            ResolveOptions::default(),
        )
        .unwrap_err();
        assert!(
            escape.0.contains("escapes") || escape.0.contains("relative"),
            "{escape}"
        );
        let env = resolve_workspace_path(
            &settings,
            Some("fixture"),
            Some(".env.local"),
            ResolveOptions::default(),
        )
        .unwrap_err();
        assert!(env.0.contains("security policy"), "{env}");
        let ignored = resolve_workspace_path(
            &settings,
            Some("fixture"),
            Some("ignored.txt"),
            ResolveOptions::default(),
        )
        .unwrap_err();
        assert!(ignored.0.contains("security policy"), "{ignored}");
        assert_eq!(
            resolve_workspace_path(&settings, Some("nope"), None, ResolveOptions::default())
                .unwrap_err()
                .0,
            "unknown workspace: nope"
        );
        assert_eq!(
            resolve_workspace_path(&settings, None, None, ResolveOptions::default())
                .unwrap_err()
                .0,
            "unknown workspace: null"
        );

        let created = resolve_workspace_path(
            &settings,
            Some("fixture"),
            Some("src/new.txt"),
            ResolveOptions::write_file(),
        )
        .unwrap();
        assert!(!created.exists);
        write_workspace_file(&created, "new").unwrap();
        assert_eq!(
            fs::read_to_string(root.join("src").join("new.txt")).unwrap(),
            "new"
        );

        let linked = link_dir(&outside, &root.join("linked-outside"));
        if linked {
            let read = resolve_workspace_path(
                &settings,
                Some("fixture"),
                Some("linked-outside/escape.txt"),
                ResolveOptions::file(),
            )
            .unwrap_err();
            assert!(
                read.0.contains("symlink")
                    || read.0.contains("junction")
                    || read.0.contains("security policy"),
                "{read}"
            );
            let write = resolve_workspace_path(
                &settings,
                Some("fixture"),
                Some("linked-outside/new.txt"),
                ResolveOptions::write_file(),
            )
            .unwrap_err();
            assert!(
                write.0.contains("symlink")
                    || write.0.contains("junction")
                    || write.0.contains("security policy"),
                "{write}"
            );
        }
    }

    #[cfg(windows)]
    fn link_dir(target: &Path, link: &Path) -> bool {
        std::os::windows::fs::symlink_dir(target, link).is_ok()
    }

    #[cfg(unix)]
    fn link_dir(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[cfg(not(any(unix, windows)))]
    fn link_dir(_target: &Path, _link: &Path) -> bool {
        false
    }

    #[test]
    fn relative_normalisation_rejects_absolute_and_escaping_paths() {
        assert_eq!(normalize_relative(None).unwrap(), Vec::<String>::new());
        assert_eq!(normalize_relative(Some("")).unwrap(), Vec::<String>::new());
        assert_eq!(
            normalize_relative(Some("./src/../src/./a.txt")).unwrap(),
            vec!["src", "a.txt"]
        );
        assert_eq!(
            normalize_relative(Some("src/..")).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            normalize_relative(Some("..")).unwrap_err().0,
            "path escapes the workspace root"
        );
        assert_eq!(
            normalize_relative(Some("src/../../x")).unwrap_err().0,
            "path escapes the workspace root"
        );
        assert_eq!(
            normalize_relative(Some("/etc/passwd")).unwrap_err().0,
            "workspace paths must be relative"
        );
        assert_eq!(
            normalize_relative(Some("a\0b")).unwrap_err().0,
            "workspace paths must be relative"
        );
        assert_eq!(
            normalize_relative(Some("C:\\x")).unwrap_err().0,
            "workspace paths must be relative"
        );
        assert_eq!(
            normalize_relative(Some("C:x")).unwrap_err().0,
            "workspace paths must be relative"
        );
    }

    #[test]
    fn sensitive_defaults_and_ignore_validity() {
        assert!(sensitive_default("a/.ssh/config"));
        assert!(sensitive_default(".config/gcloud/creds"));
        assert!(!sensitive_default(".config/other"));
        assert!(sensitive_default("deploy/.env.production"));
        assert!(sensitive_default("certs/server.KEY"));
        assert!(!sensitive_default("src/main.rs"));
        assert!(!sensitive_default(".pem"), "a dotfile has no extension");
        assert!(is_ignore_path_valid("src/a.txt"));
        assert!(!is_ignore_path_valid("./a"));
        assert!(!is_ignore_path_valid("../a"));
        assert!(!is_ignore_path_valid("/a"));
        assert!(!is_ignore_path_valid(".."));
        assert!(!is_ignore_path_valid(""));
    }

    #[test]
    fn gitignore_rules_follow_git_semantics() {
        let rules = IgnoreRules::parse("# comment\n*.log\n!keep.log\nbuild/\n/root-only.txt\ndocs/**\n**/deep\nspace\\ name\n[ab]x\n");
        let ignored = |p: &str| rules.test(p).ignored;
        assert!(ignored("a.log"));
        assert!(ignored("x/y/a.log"));
        assert!(!ignored("keep.log"));
        assert!(ignored("build/"));
        assert!(ignored("build/x.txt"));
        assert!(!ignored("build"), "a directory rule needs a directory");
        assert!(ignored("root-only.txt"));
        assert!(!ignored("sub/root-only.txt"));
        assert!(ignored("docs/anything/here"));
        assert!(ignored("a/b/deep"));
        assert!(ignored("space name"));
        assert!(ignored("ax"));
        assert!(!ignored("cx"));
        let nested = IgnoreRules::parse("build/\n!build/keep.txt\n");
        assert!(
            nested.test("build/keep.txt").ignored,
            "parent exclusion wins"
        );
        assert!(
            !IgnoreRules::parse("*.secret\n!keep.secret\n")
                .test("n/keep.secret")
                .ignored
        );
    }

    #[test]
    fn tree_and_file_routes_shape_their_replies() {
        let fixture = tempfile::tempdir().unwrap();
        let root = fixture.path().join("ws");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(root.join("Zeta")).unwrap();
        write(&root.join("src").join("a.txt"), "alpha");
        write(&root.join("b.txt"), "bee");
        write(&root.join("secret.pem"), "x");
        let settings = settings_for(&root, fixture.path());

        let tree = fs_tree(&settings, Some("fixture"), None).unwrap();
        let names: Vec<&str> = tree["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec!["src", "Zeta", "b.txt"],
            "dirs first, then locale order; skips and secrets hidden"
        );
        assert_eq!(
            tree["entries"][2],
            json!({ "name": "b.txt", "path": "b.txt", "dir": false, "size": 3 })
        );
        assert_eq!(tree["dir"], "");
        let nested = fs_tree(&settings, Some("fixture"), Some("src")).unwrap();
        assert_eq!(nested["entries"][0]["path"], "src/a.txt");
        assert_eq!(nested["entries"][0]["size"], 5);
        // A read never allows a missing leaf: the lstat error surfaces as-is.
        assert!(fs_tree(&settings, Some("fixture"), Some("missing"))
            .unwrap_err()
            .0
            .starts_with("ENOENT"));
        // A write may target a missing file, never a missing directory.
        assert_eq!(
            resolve_workspace_path(
                &settings,
                Some("fixture"),
                Some("missing"),
                ResolveOptions {
                    operation: Operation::Write,
                    target: TargetType::Directory,
                    allow_sensitive: false
                }
            )
            .unwrap_err()
            .0,
            "workspace directory does not exist"
        );
        assert_eq!(
            fs_tree(&settings, Some("fixture"), Some("b.txt"))
                .unwrap_err()
                .0,
            "workspace target is not a directory"
        );

        let file = fs_file(&settings, Some("fixture"), Some("src/a.txt")).unwrap();
        assert_eq!(
            file,
            json!({ "ws": "fixture", "path": "src/a.txt", "size": 5, "tooLarge": false, "content": "alpha" })
        );
        assert_eq!(
            fs_file(&settings, Some("fixture"), Some("src"))
                .unwrap_err()
                .0,
            "workspace target is not a regular file"
        );

        let put = fs_put_file(
            &settings,
            &json!({ "ws": "fixture", "path": "src/new.txt", "content": "héllo" }),
        )
        .unwrap();
        assert_eq!(put, json!({ "ok": true, "bytes": 6 }));
        assert_eq!(
            fs::read_to_string(root.join("src").join("new.txt")).unwrap(),
            "héllo"
        );
        let empty = fs_put_file(
            &settings,
            &json!({ "ws": "fixture", "path": "src/new.txt" }),
        )
        .unwrap();
        assert_eq!(empty["bytes"], 0);
        assert_eq!(
            fs_put_file(
                &settings,
                &json!({ "ws": "fixture", "path": "../x.txt", "content": "" })
            )
            .unwrap_err()
            .0,
            "path escapes the workspace root"
        );
        assert!(fs_put_file(
            &settings,
            &json!({ "ws": "fixture", "path": "nope/x.txt", "content": "" })
        )
        .unwrap_err()
        .0
        .starts_with("ENOENT"));
        assert_eq!(
            fs_put_file(
                &settings,
                &json!({ "ws": "fixture", "path": "src", "content": "" })
            )
            .unwrap_err()
            .0,
            "workspace target is not a regular file"
        );
    }

    #[test]
    fn posix_join_normalises() {
        assert_eq!(posix_join("", "name"), "name");
        assert_eq!(posix_join("src/", "x"), "src/x");
        assert_eq!(posix_join("./src", "x"), "src/x");
        assert_eq!(posix_join("a\\b".replace('\\', "/").as_str(), "c"), "a/b/c");
        assert_eq!(posix_join("", ""), ".");
    }
}
