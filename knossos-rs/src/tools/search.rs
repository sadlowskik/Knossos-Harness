//! Content search across the workspace, honouring .gitignore.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use serde_json::json;

use super::{req_str, Tool, ToolCtx, ToolOutput};
use crate::mnemosyne::Mnemosyne;

const MAX_RESULTS: usize = 200;
/// Cap on how much retrieved source one call may return.
const MAX_RETRIEVAL_BYTES: usize = 16_000;

pub struct Search;

#[async_trait]
impl Tool for Search {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Regex search over workspace file contents. Respects .gitignore. Returns path:line: matched text."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Rust regex syntax"},
                "glob": {"type": "string", "description": "Restrict to matching files, e.g. *.rs"},
                "max_results": {"type": "integer"}
            },
            "required": ["pattern"]
        })
    }

    fn consequential(&self) -> bool {
        false
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let pattern = req_str(input, "pattern")?;
        let re = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutput::error(format!("invalid regex: {e}"))),
        };
        let limit = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(MAX_RESULTS as u64) as usize;

        let root = ctx.root().to_path_buf();
        let glob = input.get("glob").and_then(|v| v.as_str()).map(String::from);

        // Walking a tree is blocking work; keep it off the async runtime.
        let hits = tokio::task::spawn_blocking(move || search_tree(&root, &re, glob.as_deref(), limit))
            .await??;

        if hits.is_empty() {
            return Ok(ToolOutput::ok("no matches"));
        }
        let truncated = hits.len() >= limit;
        let mut body = hits.join("\n");
        if truncated {
            body.push_str(&format!("\n\n[stopped at {limit} matches — narrow the pattern]"));
        }
        Ok(ToolOutput::ok(body))
    }
}

fn search_tree(
    root: &std::path::Path,
    re: &regex::Regex,
    glob: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false).git_ignore(true);

    if let Some(g) = glob {
        let mut ob = OverrideBuilder::new(root);
        ob.add(g)?;
        builder.overrides(ob.build()?);
    }

    let mut hits = Vec::new();
    for entry in builder.build().flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        // Non-UTF8 files are binaries for our purposes; skipping is correct.
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                hits.push(format!("{}:{}: {}", rel.display(), i + 1, line.trim()));
                if hits.len() >= limit {
                    return Ok(hits);
                }
            }
        }
    }
    Ok(hits)
}

/// Retrieval over the Mnemosyne index — the fuzzy counterpart to Scribe.
///
/// `search` answers "which lines match this regex". This answers "which parts
/// of the codebase are about this", which is the question an agent actually
/// needs when it does not yet know what to grep for.
pub struct SearchCode {
    index: Arc<Mnemosyne>,
}

impl SearchCode {
    pub fn new(index: Arc<Mnemosyne>) -> Self {
        SearchCode { index }
    }
}

#[async_trait]
impl Tool for SearchCode {
    fn name(&self) -> &str {
        "search_code"
    }

    fn description(&self) -> &str {
        "Find the parts of the codebase relevant to a description, ranked. Use this when you do \
         not know which file to look in — it matches meaning-by-wording rather than an exact \
         pattern. Use `search` instead when you know the exact string or regex."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are looking for, e.g. 'where tokens are verified'"
                },
                "limit": {"type": "integer", "description": "Maximum chunks to return (default 5)"}
            },
            "required": ["query"]
        })
    }

    fn consequential(&self) -> bool {
        false
    }

    async fn run(&self, input: &serde_json::Value, _ctx: &ToolCtx) -> Result<ToolOutput> {
        let query = req_str(input, "query")?;
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        if self.index.is_empty() {
            return Ok(ToolOutput::ok("the code index is empty"));
        }

        let hits = self.index.search(query, limit.clamp(1, 20));
        if hits.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "nothing relevant to `{query}` ({} chunks indexed)",
                self.index.chunk_count()
            )));
        }

        let mut body = String::new();
        for hit in hits {
            let block = format!("\n## {}\n```\n{}\n```\n", hit.chunk.location(), hit.chunk.text);
            if body.len() + block.len() > MAX_RETRIEVAL_BYTES {
                body.push_str("\n[further results omitted for length]\n");
                break;
            }
            body.push_str(&block);
        }
        Ok(ToolOutput::ok(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn code_search_finds_the_relevant_region() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(
            root.join("auth.rs"),
            "pub fn verify_token(t: &str) -> bool { !t.is_empty() }\n",
        )
        .unwrap();
        std::fs::write(root.join("math.rs"), "pub fn add(a: u32) -> u32 { a }\n").unwrap();

        let index = Arc::new(
            Mnemosyne::build(&root, &crate::scribe::RustAdapter).unwrap(),
        );
        let tool = SearchCode::new(index);
        let ctx = ToolCtx::new(root);

        let out = tool
            .run(&json!({"query": "verifying a token"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("verify_token"), "got: {}", out.content);
    }

    #[tokio::test]
    async fn code_search_says_so_when_nothing_matches() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("a.rs"), "pub fn add(a: u32) -> u32 { a }\n").unwrap();

        let index = Arc::new(Mnemosyne::build(&root, &crate::scribe::RustAdapter).unwrap());
        let out = SearchCode::new(index)
            .run(&json!({"query": "kubernetes ingress"}), &ToolCtx::new(root))
            .await
            .unwrap();
        assert!(out.content.contains("nothing relevant"));
    }

    #[tokio::test]
    async fn finds_matches_and_reports_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        let ctx = ToolCtx::new(root);

        let out = Search
            .run(&json!({"pattern": "fn beta"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("a.rs:2:"), "got: {}", out.content);
    }

    #[tokio::test]
    async fn glob_restricts_the_file_set() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("a.rs"), "target\n").unwrap();
        std::fs::write(root.join("b.txt"), "target\n").unwrap();
        let ctx = ToolCtx::new(root);

        let out = Search
            .run(&json!({"pattern": "target", "glob": "*.rs"}), &ctx)
            .await
            .unwrap();
        assert!(out.content.contains("a.rs"));
        assert!(!out.content.contains("b.txt"));
    }

    #[tokio::test]
    async fn invalid_regex_is_reported_not_panicked() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let ctx = ToolCtx::new(root);
        let out = Search.run(&json!({"pattern": "("}), &ctx).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("invalid regex"));
    }
}
