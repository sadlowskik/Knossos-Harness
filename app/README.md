# Knossos desktop

The Field operator surface as a desktop app, on the Rust harness. The app
links the `knossos` crate, starts the Field server in-process on a
loopback port the OS picks, and opens its window on the one-time
bootstrap link, which mints the session cookie exactly as a browser
would. No Node anywhere.

Each agent session is a `knossos serve` child process, so the harness
binary must be reachable: the app looks at `FIELD_KNOSSOS_BIN`, then for
`knossos` (or `knossos.exe`) beside its own executable, then in a sibling
checkout's `target/` directory, then on `PATH`. Installers stage the
harness beside the app as a sidecar.

Endpoint keys added through the Models screen go to the macOS Keychain
or the Windows Credential Manager; Linux keeps a 0600 `keys.json` in the
state directory until the Secret Service backend lands. Set
`FIELD_KEYCHAIN=off` to keep the file everywhere (portable installs).

Atlas is the default face: projects, agents, models, approvals, in plain
words. Rome, the RTS map with its Roman vocabulary, is the operator mode
behind the theme switch. Both are projections over the same event log.

## Run from a checkout

Build the web client once, then run the app:

```bash
npm --prefix ../field/web run build
```

```bash
cargo run
```

The app finds `field/field.yaml` and `field/web/dist` beside the crate.
Set `KNOSSOS_FIELD_DIR` to a directory holding `field/field.yaml` and
`web/dist` to point it elsewhere. Durable state (the event log) lives in
the platform app-data directory under `field-state`, or wherever
`FIELD_STATE` points.

## Build installers

```bash
cargo tauri build
```

Bundles `field/field` and `field/web/dist` as resources. Installers are
unsigned until the signing accounts exist; that is `KNS-APP-001` step 6 in
the roadmap.

## What works, what does not yet

The Rust server serves everything the map, the boards and the history
read: state, world, config, events, trace, campaign trace and replay,
positions, control groups, the WebSocket feed. Routes that still depend
on the Node harness registry (starting agents, campaigns, routines, the
file browser, git views, the terminal) answer `501 not_ported` until their
port lands, so the app is a viewer over an existing event log until the
adapter port. Run the Node server (`knossos field`) for those today.

## Requirements

- Windows: WebView2 (present on Windows 11 and recent Windows 10).
- macOS: 10.15 or newer.
- Linux: `webkit2gtk-4.1` and `libayatana-appindicator3` for the tray.

The workspace jail for child processes exists on Linux and macOS only.
On Windows the app runs agents unconfined and says so in every result;
the confined path there is WSL2.
