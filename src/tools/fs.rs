//! Filesystem tools. Every path goes through [`ToolCtx::resolve`] first.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::{req_str, Tool, ToolCtx, ToolOutput};

/// Cap on any single tool's output, so one `read` of a generated file cannot
/// swallow the whole context window.
const MAX_OUTPUT: usize = 30_000;

fn truncate(s: String, note: &str) -> String {
    if s.len() <= MAX_OUTPUT {
        return s;
    }
    let mut cut = MAX_OUTPUT;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n\n[truncated at {MAX_OUTPUT} bytes — {note}]", &s[..cut])
}

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file from the workspace. Returns 1-indexed numbered lines. Use offset/limit for large files."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "offset": {"type": "integer", "description": "1-indexed first line to read"},
                "limit": {"type": "integer", "description": "Maximum number of lines"}
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let path = ctx.resolve(req_str(input, "path")?)?;
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(t) => t,
            Err(e) => return Ok(ToolOutput::error(format!("cannot read {}: {e}", ctx.display(&path)))),
        };

        let offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(2000) as usize;

        let numbered: String = text
            .lines()
            .enumerate()
            .skip(offset - 1)
            .take(limit)
            .map(|(i, l)| format!("{:>6}\t{}\n", i + 1, l))
            .collect();

        if numbered.is_empty() {
            return Ok(ToolOutput::ok(format!("{} is empty", ctx.display(&path))));
        }
        Ok(ToolOutput::ok(truncate(numbered, "use offset/limit")))
    }
}

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write a file, creating parent directories and overwriting any existing content. For partial changes prefer edit_file."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        })
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let path = ctx.resolve(req_str(input, "path")?)?;
        let content = req_str(input, "content")?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, content).await?;

        let n = content.lines().count();
        Ok(ToolOutput::ok(format!("wrote {} ({n} lines)", ctx.display(&path)))
            .changed(path))
    }
}

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace an exact string in a file. old_string must appear exactly once — include surrounding context to make it unique."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string", "description": "Exact text to replace, including indentation"},
                "new_string": {"type": "string", "description": "Replacement text"}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let path = ctx.resolve(req_str(input, "path")?)?;
        let old = req_str(input, "old_string")?;
        let new = req_str(input, "new_string")?;

        let text = match tokio::fs::read_to_string(&path).await {
            Ok(t) => t,
            Err(e) => return Ok(ToolOutput::error(format!("cannot read {}: {e}", ctx.display(&path)))),
        };

        // Uniqueness is enforced rather than assumed: a silent multi-replace is
        // how an edit tool corrupts a file in a way nobody notices for hours.
        match text.matches(old).count() {
            0 => Ok(ToolOutput::error(format!(
                "old_string not found in {}",
                ctx.display(&path)
            ))),
            1 => {
                tokio::fs::write(&path, text.replacen(old, new, 1)).await?;
                Ok(ToolOutput::ok(format!("edited {}", ctx.display(&path))).changed(path))
            }
            n => Ok(ToolOutput::error(format!(
                "old_string appears {n} times in {}; add surrounding context to make it unique",
                ctx.display(&path)
            ))),
        }
    }
}

pub struct ListDir;

#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the entries of a directory. Directories are suffixed with /."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Defaults to the workspace root"}}
        })
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let requested = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let path = ctx.resolve(requested)?;

        let mut entries = match tokio::fs::read_dir(&path).await {
            Ok(e) => e,
            Err(e) => return Ok(ToolOutput::error(format!("cannot list {}: {e}", ctx.display(&path)))),
        };

        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let name = entry.file_name().to_string_lossy().to_string();
            names.push(if is_dir { format!("{name}/") } else { name });
        }
        names.sort();

        if names.is_empty() {
            return Ok(ToolOutput::ok(format!("{} is empty", ctx.display(&path))));
        }
        Ok(ToolOutput::ok(truncate(names.join("\n"), "narrow the path")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        (dir, ToolCtx::new(root))
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let (_d, c) = ctx();
        let w = WriteFile
            .run(&json!({"path": "a/b.rs", "content": "fn main() {}\n"}), &c)
            .await
            .unwrap();
        assert!(!w.is_error);
        assert_eq!(w.changed.len(), 1);

        let r = ReadFile.run(&json!({"path": "a/b.rs"}), &c).await.unwrap();
        assert!(r.content.contains("fn main() {}"));
        assert!(r.content.contains("     1\t"));
    }

    #[tokio::test]
    async fn edit_refuses_ambiguous_matches() {
        let (_d, c) = ctx();
        WriteFile
            .run(&json!({"path": "d.rs", "content": "let x = 1;\nlet x = 1;\n"}), &c)
            .await
            .unwrap();
        let e = EditFile
            .run(&json!({"path": "d.rs", "old_string": "let x = 1;", "new_string": "let y = 2;"}), &c)
            .await
            .unwrap();
        assert!(e.is_error);
        assert!(e.content.contains("appears 2 times"));
    }

    #[tokio::test]
    async fn edit_applies_a_unique_match() {
        let (_d, c) = ctx();
        WriteFile
            .run(&json!({"path": "d.rs", "content": "let x = 1;\n"}), &c)
            .await
            .unwrap();
        let e = EditFile
            .run(&json!({"path": "d.rs", "old_string": "let x = 1;", "new_string": "let y = 2;"}), &c)
            .await
            .unwrap();
        assert!(!e.is_error, "{}", e.content);
        let after = std::fs::read_to_string(c.root().join("d.rs")).unwrap();
        assert_eq!(after, "let y = 2;\n");
    }

    #[tokio::test]
    async fn writing_outside_the_jail_is_refused() {
        let (_d, c) = ctx();
        let out = WriteFile
            .run(&json!({"path": "../escaped.rs", "content": "x"}), &c)
            .await;
        assert!(out.is_err(), "path jail must reject ../ writes");
    }
}
