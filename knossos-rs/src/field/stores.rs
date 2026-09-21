//! Local stores under the state directory. Port of
//! `field/server/src/keystore.js` and `endpoints-store.js`.
//!
//! Key values live only here, are registered for event-log redaction, and
//! are never written to `field.yaml`, the snapshot or any API response. The
//! API exposes key *ids*, never values. Endpoint descriptors added through
//! the UI are non-secret and reference a key by id.

use super::js::get_str;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct KeyStore {
    file: PathBuf,
    map: BTreeMap<String, String>,
}

impl KeyStore {
    pub fn open(state_dir: &Path) -> KeyStore {
        let file = state_dir.join("keys.json");
        let map = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|v| v.as_object().cloned())
            .map(|obj| {
                obj.into_iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        KeyStore { file, map }
    }

    fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string(&self.map).unwrap_or_else(|_| "{}".into());
        write_private(&self.file, &text)
    }

    pub fn set(&mut self, id: &str, value: &str) -> std::io::Result<()> {
        self.map.insert(id.to_string(), value.to_string());
        self.save()
    }

    pub fn get(&self, id: &str) -> Option<&str> {
        self.map.get(id).map(String::as_str)
    }

    pub fn delete(&mut self, id: &str) -> std::io::Result<bool> {
        let removed = self.map.remove(id).is_some();
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    pub fn ids(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<String> {
        self.map.values().cloned().collect()
    }
}

/// Writes a file the owner alone can read (0600 on Unix; Windows relies on
/// the per-user profile ACL of the state directory).
fn write_private(file: &Path, text: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(file)?;
        f.write_all(text.as_bytes())
    }
    #[cfg(not(unix))]
    std::fs::write(file, text)
}

#[derive(Debug)]
pub struct EndpointsStore {
    file: PathBuf,
    list: Vec<Value>,
}

impl EndpointsStore {
    pub fn open(state_dir: &Path) -> EndpointsStore {
        let file = state_dir.join("endpoints.json");
        let list = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|v| v.get("endpoints").and_then(Value::as_array).cloned())
            .unwrap_or_default();
        EndpointsStore { file, list }
    }

    fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string_pretty(
            &json!({ "schema": "field-endpoints/v1", "endpoints": self.list }),
        )
        .unwrap_or_else(|_| "{}".into());
        std::fs::write(&self.file, text)
    }

    pub fn all(&self) -> Vec<Value> {
        self.list.clone()
    }

    pub fn get(&self, id: &str) -> Option<&Value> {
        self.list.iter().find(|e| get_str(e, "id") == Some(id))
    }

    pub fn add(&mut self, desc: Value) -> std::io::Result<()> {
        let id = get_str(&desc, "id").unwrap_or("").to_string();
        self.list.retain(|e| get_str(e, "id") != Some(&id));
        self.list.push(desc);
        self.save()
    }

    pub fn remove(&mut self, id: &str) -> std::io::Result<bool> {
        let before = self.list.len();
        self.list.retain(|e| get_str(e, "id") != Some(id));
        let removed = self.list.len() != before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_endpoints_survive_a_reopen_and_never_mix() {
        let dir = tempfile::tempdir().unwrap();
        let mut keys = KeyStore::open(dir.path());
        keys.set("ep:local", "sk-secret-value").unwrap();
        let mut endpoints = EndpointsStore::open(dir.path());
        endpoints
            .add(json!({ "id": "local", "name": "Local", "kind": "openai-compatible", "secretRef": "ep:local" }))
            .unwrap();
        endpoints
            .add(json!({ "id": "local", "name": "Local again", "kind": "openai-compatible" }))
            .unwrap();

        let keys = KeyStore::open(dir.path());
        assert_eq!(keys.get("ep:local"), Some("sk-secret-value"));
        let endpoints = EndpointsStore::open(dir.path());
        assert_eq!(endpoints.all().len(), 1, "re-adding replaces by id");
        assert_eq!(endpoints.get("local").unwrap()["name"], "Local again");
        let on_disk = std::fs::read_to_string(dir.path().join("endpoints.json")).unwrap();
        assert!(
            !on_disk.contains("sk-secret-value"),
            "the endpoint file never holds a key"
        );
    }
}
