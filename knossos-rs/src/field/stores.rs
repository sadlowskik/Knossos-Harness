//! Local stores under the state directory. Port of
//! `field/server/src/keystore.js` and `endpoints-store.js`, with the key
//! values moved into the operating system's credential store where one
//! exists.
//!
//! Key values are registered for event-log redaction, never written to
//! `field.yaml`, the snapshot or any API response. The API exposes key
//! *ids*, never values. Endpoint descriptors added through the UI are
//! non-secret and reference a key by id.
//!
//! Backends:
//! - **keychain** (macOS Keychain, Windows Credential Manager): `keys.json`
//!   holds only the ids; each value lives under the `knossos-field` service.
//! - **file**: `keys.json` holds the values too, mode 0600 on Unix. Used on
//!   Linux (no Secret Service dependency yet), when `FIELD_KEYCHAIN=off`, in
//!   tests, and whenever the credential store refuses a write.

use super::js::get_str;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "field-keys/v2";
#[cfg_attr(not(any(target_os = "macos", windows)), allow(dead_code))]
const SERVICE: &str = "knossos-field";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyBackend {
    Keychain,
    File,
}

impl KeyBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            KeyBackend::Keychain => "keychain",
            KeyBackend::File => "file",
        }
    }
}

#[derive(Debug)]
pub struct KeyStore {
    file: PathBuf,
    backend: KeyBackend,
    /// Every stored id, whichever backend holds the value.
    ids: BTreeSet<String>,
    /// Values the keychain does not hold (file backend, or a keychain write
    /// that failed). Persisted.
    fallback: BTreeMap<String, String>,
    /// Values read from the keychain, cached for the process lifetime.
    cache: BTreeMap<String, String>,
}

impl KeyStore {
    /// The file backend only. Tests and portable installs use this; it never
    /// touches the operating system's credential store.
    pub fn open(state_dir: &Path) -> KeyStore {
        Self::open_with(state_dir, false)
    }

    /// The keychain when the platform has one and `FIELD_KEYCHAIN` is not
    /// `off`; the file otherwise.
    pub fn open_default(state_dir: &Path) -> KeyStore {
        let wanted = !matches!(
            std::env::var("FIELD_KEYCHAIN").as_deref(),
            Ok("off") | Ok("0") | Ok("false")
        );
        Self::open_with(state_dir, wanted && keychain::available())
    }

    fn open_with(state_dir: &Path, keychain: bool) -> KeyStore {
        let file = state_dir.join("keys.json");
        let mut ids = BTreeSet::new();
        let mut fallback = BTreeMap::new();
        if let Some(doc) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        {
            if doc.get("schema").and_then(Value::as_str) == Some(SCHEMA) {
                for id in doc
                    .get("ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(id) = id.as_str() {
                        ids.insert(id.to_string());
                    }
                }
                if let Some(values) = doc.get("values").and_then(Value::as_object) {
                    for (k, v) in values {
                        if let Some(v) = v.as_str() {
                            ids.insert(k.clone());
                            fallback.insert(k.clone(), v.to_string());
                        }
                    }
                }
            } else if let Some(map) = doc.as_object() {
                // v1: a plain id -> value map.
                for (k, v) in map {
                    if let Some(v) = v.as_str() {
                        ids.insert(k.clone());
                        fallback.insert(k.clone(), v.to_string());
                    }
                }
            }
        }
        let mut store = KeyStore {
            file,
            backend: if keychain {
                KeyBackend::Keychain
            } else {
                KeyBackend::File
            },
            ids,
            fallback,
            cache: BTreeMap::new(),
        };
        if keychain {
            store.load_keychain();
        }
        store
    }

    /// Reads every id's value from the keychain and moves any value the file
    /// still holds into the keychain (a v1 file, or an earlier failed write).
    fn load_keychain(&mut self) {
        let mut changed = false;
        for id in self.ids.clone() {
            if let Some(value) = self.fallback.get(&id).cloned() {
                if keychain::set(&id, &value) {
                    self.fallback.remove(&id);
                    self.cache.insert(id, value);
                    changed = true;
                }
                continue;
            }
            match keychain::get(&id) {
                Some(value) => {
                    self.cache.insert(id, value);
                }
                None => {
                    // The id is on record but the credential is gone (another
                    // user profile, a wiped keychain). Forget it so the UI
                    // shows the endpoint as keyless instead of failing later.
                    self.ids.remove(&id);
                    changed = true;
                }
            }
        }
        if changed {
            let _ = self.save();
        }
    }

    pub fn backend(&self) -> KeyBackend {
        self.backend
    }

    fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let doc = json!({ "schema": SCHEMA, "ids": self.ids, "values": self.fallback });
        let text = serde_json::to_string(&doc).unwrap_or_else(|_| "{}".into());
        write_private(&self.file, &text)
    }

    pub fn set(&mut self, id: &str, value: &str) -> std::io::Result<()> {
        if self.backend == KeyBackend::Keychain && keychain::set(id, value) {
            self.fallback.remove(id);
            self.cache.insert(id.to_string(), value.to_string());
        } else {
            self.cache.remove(id);
            self.fallback.insert(id.to_string(), value.to_string());
        }
        self.ids.insert(id.to_string());
        self.save()
    }

    pub fn get(&self, id: &str) -> Option<&str> {
        self.cache
            .get(id)
            .or_else(|| self.fallback.get(id))
            .map(String::as_str)
    }

    pub fn delete(&mut self, id: &str) -> std::io::Result<bool> {
        let removed = self.ids.remove(id);
        if removed {
            if self.backend == KeyBackend::Keychain {
                keychain::delete(id);
            }
            self.cache.remove(id);
            self.fallback.remove(id);
            self.save()?;
        }
        Ok(removed)
    }

    pub fn ids(&self) -> Vec<String> {
        self.ids.iter().cloned().collect()
    }

    pub fn values(&self) -> Vec<String> {
        self.cache
            .values()
            .chain(self.fallback.values())
            .cloned()
            .collect()
    }
}

/// The operating system credential store, behind one narrow door.
mod keychain {
    #[cfg(any(target_os = "macos", windows))]
    fn ready() -> bool {
        use std::sync::OnceLock;
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| {
            #[cfg(target_os = "macos")]
            let store = apple_native_keyring_store::keychain::Store::new();
            #[cfg(windows)]
            let store = windows_native_keyring_store::Store::new();
            match store {
                Ok(store) => {
                    keyring_core::set_default_store(store);
                    true
                }
                Err(_) => false,
            }
        })
    }

    #[cfg(any(target_os = "macos", windows))]
    fn entry(id: &str) -> Option<keyring_core::Entry> {
        ready()
            .then(|| keyring_core::Entry::new(super::SERVICE, id).ok())
            .flatten()
    }

    #[cfg(any(target_os = "macos", windows))]
    pub fn available() -> bool {
        ready()
    }
    #[cfg(any(target_os = "macos", windows))]
    pub fn get(id: &str) -> Option<String> {
        entry(id)?.get_password().ok()
    }
    #[cfg(any(target_os = "macos", windows))]
    pub fn set(id: &str, value: &str) -> bool {
        entry(id).is_some_and(|e| e.set_password(value).is_ok())
    }
    #[cfg(any(target_os = "macos", windows))]
    pub fn delete(id: &str) {
        if let Some(e) = entry(id) {
            let _ = e.delete_credential();
        }
    }

    #[cfg(not(any(target_os = "macos", windows)))]
    pub fn available() -> bool {
        false
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    pub fn get(_id: &str) -> Option<String> {
        None
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    pub fn set(_id: &str, _value: &str) -> bool {
        false
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    pub fn delete(_id: &str) {}
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
        assert_eq!(keys.backend(), KeyBackend::File);
        assert_eq!(keys.get("ep:local"), Some("sk-secret-value"));
        assert_eq!(keys.ids(), vec!["ep:local".to_string()]);
        let endpoints = EndpointsStore::open(dir.path());
        assert_eq!(endpoints.all().len(), 1, "re-adding replaces by id");
        assert_eq!(endpoints.get("local").unwrap()["name"], "Local again");
        let on_disk = std::fs::read_to_string(dir.path().join("endpoints.json")).unwrap();
        assert!(
            !on_disk.contains("sk-secret-value"),
            "the endpoint file never holds a key"
        );
    }

    #[test]
    fn a_v1_key_file_is_read_and_rewritten_as_v2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("keys.json"),
            r#"{"ep:a":"value-a","ep:b":"value-b"}"#,
        )
        .unwrap();
        let mut keys = KeyStore::open(dir.path());
        assert_eq!(keys.get("ep:b"), Some("value-b"));
        assert!(keys.delete("ep:a").unwrap());
        assert!(!keys.delete("ep:a").unwrap());
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("keys.json")).unwrap())
                .unwrap();
        assert_eq!(doc["schema"], SCHEMA);
        assert_eq!(doc["ids"], json!(["ep:b"]));
        assert_eq!(doc["values"]["ep:b"], "value-b");
        let mut sorted = KeyStore::open(dir.path()).values();
        sorted.sort();
        assert_eq!(sorted, vec!["value-b".to_string()]);
    }
}
