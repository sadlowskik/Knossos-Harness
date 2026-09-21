//! Knossos desktop: a Tauri shell around the `knossos` crate.
//!
//! One process. The Field server (event log, projection, API, WebSocket)
//! starts in-process on a loopback port chosen by the OS, and the window
//! opens on the one-time bootstrap link, which mints the browser session
//! cookie the way it would in any browser. No sidecar, no Node. Closing the
//! window stops the server; the event log is the durable state.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use knossos::field::{Running, ServerOptions};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

/// Where the Field bundle lives: `field/` (configuration) and `web/dist`.
struct Layout {
    field_dir: PathBuf,
    dist_dir: PathBuf,
}

fn layout(app: &tauri::AppHandle) -> Result<Layout> {
    if let Some(root) = std::env::var_os("KNOSSOS_FIELD_DIR").filter(|v| !v.is_empty()) {
        let root = PathBuf::from(root);
        return check(&root.join("field"), &root.join("web").join("dist"))
            .with_context(|| format!("KNOSSOS_FIELD_DIR does not hold a Field bundle: {}", root.display()));
    }
    // A development checkout: the app crate beside `field/`.
    let executable = std::env::current_exe().context("cannot locate the app executable")?;
    for ancestor in executable.ancestors().take(8) {
        let candidate = ancestor.join("field");
        if let Ok(found) = check(&candidate.join("field"), &candidate.join("web").join("dist")) {
            return Ok(found);
        }
    }
    // An installed bundle: the resources Tauri copied beside the binary.
    if let Ok(resources) = app.path().resource_dir() {
        let root = resources.join("field");
        if let Ok(found) = check(&root.join("field"), &root.join("web").join("dist")) {
            return Ok(found);
        }
    }
    bail!("no Field bundle found; set KNOSSOS_FIELD_DIR to a directory holding field/field.yaml and web/dist")
}

fn check(field_dir: &Path, dist_dir: &Path) -> Result<Layout> {
    if !field_dir.join("field.yaml").is_file() {
        bail!("missing {}", field_dir.join("field.yaml").display());
    }
    if !dist_dir.join("index.html").is_file() {
        bail!("missing {}", dist_dir.join("index.html").display());
    }
    Ok(Layout {
        field_dir: field_dir.to_path_buf(),
        dist_dir: dist_dir.to_path_buf(),
    })
}

/// Durable state lives in the platform's app-data directory, never beside
/// the binary.
fn state_dir(app: &tauri::AppHandle) -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("FIELD_STATE").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let base = app.path().app_data_dir().context("no app data directory on this platform")?;
    Ok(base.join("field-state"))
}

struct Server(Mutex<Option<Running>>);

fn main() -> Result<()> {
    tauri::Builder::default()
        .manage(Server(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();
            let layout = layout(&handle)?;
            let state_dir = state_dir(&handle)?;
            let options = ServerOptions {
                field_dir: layout.field_dir,
                state_dir,
                dist_dir: layout.dist_dir,
                port: Some(0),
                ui_origin: None,
                backend: None,
                bootstrap_token: None,
                browser_token: None,
                probe_endpoints: true,
                watch_git: true,
            };
            let running = tauri::async_runtime::block_on(knossos::field::server::start(options))
                .context("starting the Field server")?;
            let url = running.bootstrap_url.clone();
            if let Some(server) = app.try_state::<Server>() {
                if let Ok(mut slot) = server.0.lock() {
                    *slot = Some(running);
                }
            }
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
                .title("Knossos")
                .inner_size(1280.0, 820.0)
                .min_inner_size(900.0, 600.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .context("building the Knossos app")?
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(server) = app.try_state::<Server>() {
                    let running = server.0.lock().ok().and_then(|mut slot| slot.take());
                    if let Some(running) = running {
                        tauri::async_runtime::block_on(running.stop());
                    }
                }
            }
        });
    Ok(())
}
