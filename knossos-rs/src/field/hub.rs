//! One socket per operator window. Port of `field/server/src/ws.js`.
//!
//! Events go out immediately (they drive the trace tail and the route
//! pulses); the folded snapshot is coalesced so a burst of tool calls
//! cannot flood the client. A client that falls too far behind is dropped
//! rather than buffered without bound.

use super::eventlog::Event;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;

pub const SNAPSHOT_INTERVAL_MS: u64 = 120;
/// Messages a client may lag behind before it is terminated; the analogue of
/// the reference's one-megabyte buffered-bytes cap.
pub const CLIENT_BACKLOG: usize = 2048;

/// What goes to every connected client.
#[derive(Debug, Clone)]
pub enum Outbound {
    Text(Arc<str>),
    /// Logout: close every socket.
    Revoke,
}

pub type SnapshotFn = Arc<dyn Fn() -> Value + Send + Sync>;

pub struct Hub {
    tx: broadcast::Sender<Outbound>,
    dirty: Arc<AtomicBool>,
    timer_armed: Arc<AtomicBool>,
    snapshot: SnapshotFn,
    interval: Duration,
    _keep: Mutex<()>,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hub")
            .field("clients", &self.clients())
            .finish()
    }
}

impl Hub {
    pub fn new(snapshot: SnapshotFn) -> Arc<Hub> {
        Self::with_interval(snapshot, Duration::from_millis(SNAPSHOT_INTERVAL_MS))
    }

    pub fn with_interval(snapshot: SnapshotFn, interval: Duration) -> Arc<Hub> {
        let (tx, _) = broadcast::channel(CLIENT_BACKLOG);
        Arc::new(Hub {
            tx,
            dirty: Arc::new(AtomicBool::new(false)),
            timer_armed: Arc::new(AtomicBool::new(false)),
            snapshot,
            interval,
            _keep: Mutex::new(()),
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Outbound> {
        self.tx.subscribe()
    }

    /// The snapshot message a client receives on connect.
    pub fn snapshot_message(&self) -> String {
        json!({ "type": "snapshot", "state": (self.snapshot)() }).to_string()
    }

    pub fn clients(&self) -> usize {
        self.tx.receiver_count()
    }

    fn send_all(&self, text: String) {
        // No receivers is not an error: nobody is watching.
        let _ = self.tx.send(Outbound::Text(Arc::from(text)));
    }

    /// Fan an appended event out at once and schedule one coalesced snapshot.
    pub fn push_event(self: &Arc<Self>, event: &Event) {
        self.send_all(json!({ "type": "event", "event": event }).to_string());
        self.dirty.store(true, Ordering::SeqCst);
        if !self.timer_armed.swap(true, Ordering::SeqCst) {
            let hub = Arc::clone(self);
            tokio::spawn(async move {
                tokio::time::sleep(hub.interval).await;
                hub.flush();
            });
        }
    }

    fn flush(&self) {
        self.timer_armed.store(false, Ordering::SeqCst);
        if !self.dirty.swap(false, Ordering::SeqCst) {
            return;
        }
        self.send_all(self.snapshot_message());
    }

    /// Any other message, e.g. terminal output.
    pub fn broadcast(&self, payload: &Value) {
        self.send_all(payload.to_string());
    }

    pub fn revoke_clients(&self) {
        let _ = self.tx.send(Outbound::Revoke);
    }
}
