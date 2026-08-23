//! A bidirectional JSON-RPC 2.0 peer over newline-delimited stdio.
//!
//! ACP is symmetric: the editor calls the agent (`session/prompt`), and the
//! agent calls the editor back mid-turn (`fs/read_text_file`,
//! `session/request_permission`). That symmetry is the only hard part of the
//! transport, and it is what forces the threading below.
//!
//! | Thread | Owns |
//! |---|---|
//! | reader | stdin. Parses one message per line. Responses resolve a waiting request; requests and notifications go on a queue. |
//! | worker | drains that queue and runs handlers. A handler may call [`PeerHandle::request`], which blocks on a `Pending` that the *reader* resolves — so it cannot deadlock against itself. |
//! | any | may write, serialised by the output mutex. |
//!
//! Doing this on one thread would deadlock the moment a handler called back
//! into the client: the reader would be sitting inside the handler, so the reply
//! it was waiting for could never be read.
//!
//! Framing rules, which are not negotiable:
//!
//! * one JSON value per line, `\n`-terminated, no embedded newlines
//! * the output stream carries protocol traffic and nothing else — a stray
//!   `println!` corrupts the stream and the editor drops the connection
//! * logging goes to stderr, which the client may show as agent logs
//!
//! # Why the peer is split in two
//!
//! Python let a handler close over the `Peer` it was installed on. That cycle
//! does not typecheck here, and papering over it with `Arc<Mutex<Peer>>` would
//! reintroduce exactly the deadlock the threading exists to avoid — the worker
//! would hold the peer lock while a handler waited on a reply the reader needs
//! that same lock to deliver.
//!
//! So the conversation is [`PeerHandle`] (cheap to clone, safe to capture, all
//! the outbound calls) and the thread ownership is [`Peer`]. A handler is handed
//! a handle it captured before the peer started, which is a compile-time
//! statement that no handler can reach the reader's state.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// An error to return to the peer, or one received from it.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        RpcError {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(code: i64, message: impl Into<String>, data: Value) -> Self {
        RpcError {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("code".into(), json!(self.code));
        out.insert("message".into(), json!(self.message));
        if let Some(data) = &self.data {
            out.insert("data".into(), data.clone());
        }
        Value::Object(out)
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RpcError {}

/// Write to stderr. Never, ever to the protocol stream.
macro_rules! log {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

/// What a handler must satisfy: called as `(method, params, is_request)`.
///
/// Return a JSON-serialisable result for requests; return [`RpcError`] to send a
/// structured error. The return value is ignored for notifications.
pub trait Handler: Fn(&str, Option<Value>, bool) -> Result<Value, RpcError> + Send + Sync {}
impl<T> Handler for T where T: Fn(&str, Option<Value>, bool) -> Result<Value, RpcError> + Send + Sync
{}

/// Decides whether a method skips the worker queue. See [`Peer::with_fast_path`].
pub type FastPath = dyn Fn(&str) -> bool + Send + Sync;

/// One outstanding outbound request.
#[derive(Debug, Default)]
struct Pending {
    outcome: Mutex<Option<Result<Value, RpcError>>>,
    woken: Condvar,
}

impl Pending {
    fn settle(&self, outcome: Result<Value, RpcError>) {
        *self.outcome.lock().expect("pending mutex") = Some(outcome);
        self.woken.notify_all();
    }
}

/// The parts of a conversation that are safe to hand to a handler.
struct Shared {
    tx: Mutex<Box<dyn Write + Send>>,
    pending: Mutex<HashMap<String, Arc<Pending>>>,
    next_id: AtomicU64,
    closed: Arc<AtomicBool>,
    inbox: Mutex<Sender<Option<Value>>>,
}

/// One end of a JSON-RPC conversation: everything except the threads.
///
/// Cloning is cheap and shares one connection. Handlers capture this, which is
/// what lets them call back into the peer mid-turn.
#[derive(Clone)]
pub struct PeerHandle {
    shared: Arc<Shared>,
}

impl PeerHandle {
    /// How often a waiting [`request`](Self::request) looks up to check `abort`
    /// and the pipe.
    pub const WAIT_SLICE: Duration = Duration::from_millis(100);

    fn send(&self, payload: &Value) {
        let line = serde_json::to_string(payload).expect("payload is serialisable");
        debug_assert!(
            !line.contains('\n'),
            "framing violation: message contains a newline"
        );
        let mut tx = self.shared.tx.lock().expect("write mutex");
        if self.shared.closed.load(Ordering::SeqCst) {
            return;
        }
        if writeln!(tx, "{line}").and_then(|()| tx.flush()).is_err() {
            self.shared.closed.store(true, Ordering::SeqCst);
        }
    }

    /// Send a notification. No reply is expected and none is waited for.
    pub fn notify(&self, method: &str, params: Option<Value>) {
        let mut msg = Map::new();
        msg.insert("jsonrpc".into(), json!("2.0"));
        msg.insert("method".into(), json!(method));
        if let Some(params) = params {
            msg.insert("params".into(), params);
        }
        self.send(&Value::Object(msg));
    }

    /// Call the peer and block for its reply, with no deadline.
    pub fn request(&self, method: &str, params: Option<Value>) -> Result<Value, RpcError> {
        self.request_with(method, params, None, None)
    }

    /// Call the peer and block for its reply.
    ///
    /// `abort` makes an untimed wait interruptible. Some requests legitimately
    /// have no deadline — a permission prompt waits as long as the user takes to
    /// read it — but "no deadline" must not mean "unkillable": if the turn is
    /// cancelled, or the peer goes away, the caller has to get control back.
    pub fn request_with(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Option<Duration>,
        abort: Option<&Arc<AtomicBool>>,
    ) -> Result<Value, RpcError> {
        let req_id = format!(
            "h{}",
            self.shared.next_id.fetch_add(1, Ordering::SeqCst) + 1
        );
        let pending = Arc::new(Pending::default());
        self.shared
            .pending
            .lock()
            .expect("pending mutex")
            .insert(req_id.clone(), Arc::clone(&pending));

        let mut msg = Map::new();
        msg.insert("jsonrpc".into(), json!("2.0"));
        msg.insert("id".into(), json!(req_id));
        msg.insert("method".into(), json!(method));
        if let Some(params) = params {
            msg.insert("params".into(), params);
        }
        self.send(&Value::Object(msg));

        match self.wait_for(&pending, timeout, abort) {
            Some(outcome) => outcome,
            None => {
                self.shared
                    .pending
                    .lock()
                    .expect("pending mutex")
                    .remove(&req_id);
                if abort.is_some_and(|a| a.load(Ordering::SeqCst)) {
                    Err(RpcError::new(
                        INTERNAL_ERROR,
                        format!("{method} was cancelled"),
                    ))
                } else if self.shared.closed.load(Ordering::SeqCst) {
                    Err(RpcError::new(
                        INTERNAL_ERROR,
                        format!("peer closed before replying to {method}"),
                    ))
                } else {
                    Err(RpcError::new(
                        INTERNAL_ERROR,
                        format!("timed out waiting for reply to {method}"),
                    ))
                }
            }
        }
    }

    /// Wait for `pending`, giving up on timeout, abort, or a closed pipe.
    ///
    /// Woken every [`WAIT_SLICE`](Self::WAIT_SLICE) even when untimed: the wait
    /// is still bounded by the pipe, and waking to notice a dead peer is the
    /// difference between a stalled turn and a hung process.
    fn wait_for(
        &self,
        pending: &Pending,
        timeout: Option<Duration>,
        abort: Option<&Arc<AtomicBool>>,
    ) -> Option<Result<Value, RpcError>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut outcome = pending.outcome.lock().expect("pending mutex");
        loop {
            if let Some(settled) = outcome.take() {
                return Some(settled);
            }
            if abort.is_some_and(|a| a.load(Ordering::SeqCst))
                || self.shared.closed.load(Ordering::SeqCst)
            {
                return None;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let (guard, _) = pending
                .woken
                .wait_timeout(outcome, Self::WAIT_SLICE)
                .expect("pending mutex");
            outcome = guard;
        }
    }

    /// Stop the peer. Idempotent.
    pub fn close(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        let _ = self.shared.inbox.lock().expect("inbox mutex").send(None);
    }

    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }

    /// How many requests are still awaiting a reply. Abandoned ones are gone.
    pub fn pending_count(&self) -> usize {
        self.shared.pending.lock().expect("pending mutex").len()
    }
}

/// A running JSON-RPC conversation, and the threads that drive it.
pub struct Peer {
    handle: PeerHandle,
    rx: Option<Box<dyn BufRead + Send>>,
    inbox_rx: Option<Receiver<Option<Value>>>,
    /// Methods the worker queue would starve.
    ///
    /// `session/cancel` is the reason this exists: it arrives *during* a
    /// `session/prompt`, so queueing it behind that prompt means it can only be
    /// delivered once the turn it was meant to interrupt has already finished.
    /// Fast-path handlers run on the reader thread and must therefore never
    /// block.
    fast_path: Option<Box<FastPath>>,
    threads: Vec<JoinHandle<()>>,
}

impl Peer {
    pub fn new(rx: Box<dyn BufRead + Send>, tx: Box<dyn Write + Send>) -> Self {
        let (inbox_tx, inbox_rx) = channel();
        let shared = Arc::new(Shared {
            tx: Mutex::new(tx),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            closed: Arc::new(AtomicBool::new(false)),
            inbox: Mutex::new(inbox_tx),
        });
        Peer {
            handle: PeerHandle { shared },
            rx: Some(rx),
            inbox_rx: Some(inbox_rx),
            fast_path: None,
            threads: Vec::new(),
        }
    }

    /// The conversation, detached from the threads. Capture this in handlers.
    pub fn handle(&self) -> PeerHandle {
        self.handle.clone()
    }

    /// Mark methods that must not wait behind the worker queue.
    pub fn with_fast_path(mut self, f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        self.fast_path = Some(Box::new(f));
        self
    }

    /// Spawn the reader and worker threads.
    pub fn start(&mut self, handler: impl Handler + 'static) {
        let handler: Arc<dyn Handler> = Arc::new(handler);
        let rx = self.rx.take().expect("start called twice");
        let inbox_rx = self.inbox_rx.take().expect("start called twice");
        let fast_path = self.fast_path.take();

        let reader = {
            let handle = self.handle.clone();
            let handler = Arc::clone(&handler);
            std::thread::Builder::new()
                .name("acp-reader".into())
                .spawn(move || read_loop(handle, rx, handler, fast_path))
                .expect("spawn reader")
        };
        let worker = {
            let handle = self.handle.clone();
            std::thread::Builder::new()
                .name("acp-worker".into())
                .spawn(move || {
                    while let Ok(Some(msg)) = inbox_rx.recv() {
                        dispatch(&handle, &*handler, &msg);
                    }
                })
                .expect("spawn worker")
        };
        self.threads.push(reader);
        self.threads.push(worker);
    }

    /// Block until both threads finish.
    pub fn wait(&mut self) {
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }

    pub fn serve_forever(&mut self, handler: impl Handler + 'static) {
        self.start(handler);
        self.wait();
    }

    pub fn close(&self) {
        self.handle.close();
    }

    pub fn is_closed(&self) -> bool {
        self.handle.is_closed()
    }
}

fn read_loop(
    handle: PeerHandle,
    rx: Box<dyn BufRead + Send>,
    handler: Arc<dyn Handler>,
    fast_path: Option<Box<FastPath>>,
) {
    for line in rx.lines() {
        if handle.is_closed() {
            break;
        }
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // One bad message must not take the connection with it. Python learned
        // this the expensive way: `{"method": []}` reached a hash lookup on an
        // unhashable key, the `TypeError` escaped a loop that only caught
        // `(OSError, ValueError)`, and the `finally` then closed the peer and
        // stranded every pending request — so one line from the editor could
        // kill the connection and take a session's staged edits with it.
        //
        // Rust's type system rules out that particular failure, but not the
        // shape of it, so accept() reports rather than panics and a panicking
        // handler is caught below.
        accept(&handle, line, &handler, fast_path.as_deref());
    }

    handle.shared.closed.store(true, Ordering::SeqCst);
    let _ = handle.shared.inbox.lock().expect("inbox mutex").send(None);
    // Nothing will ever answer an outstanding request now.
    let stranded: Vec<Arc<Pending>> = {
        let mut pending = handle.shared.pending.lock().expect("pending mutex");
        pending.drain().map(|(_, p)| p).collect()
    };
    for p in stranded {
        p.settle(Err(RpcError::new(INTERNAL_ERROR, "connection closed")));
    }
}

/// Route one line. Anything malformed is logged and dropped.
///
/// The shape checks are not decoration. JSON-RPC says `method` is a string and
/// `id` is a string, number or null, but nothing stops a peer sending `[]` for
/// either. Rejecting them here keeps the failure at the edge, where it is one
/// dropped message rather than a dead connection.
fn accept(
    handle: &PeerHandle,
    line: &str,
    handler: &Arc<dyn Handler>,
    fast_path: Option<&FastPath>,
) {
    let msg: Value = match serde_json::from_str(line) {
        Ok(msg) => msg,
        Err(exc) => {
            log!("[jsonrpc] dropping unparseable line: {exc}");
            return;
        }
    };
    let Some(obj) = msg.as_object() else { return };

    if let Some(method) = obj.get("method") {
        let Some(method) = method.as_str() else {
            log!("[jsonrpc] dropping message whose method is not a string");
            return;
        };
        // Notifications *and* requests: `session/interject` is a request that
        // must land during `session/prompt`, the same reason `session/cancel`
        // is a fast-path notification. Dispatch is instant (set a flag / push
        // a queue); it must not wait on the worker holding the turn lock.
        if fast_path.is_some_and(|f| f(method)) {
            dispatch(handle, &**handler, &msg);
        } else {
            let _ = handle
                .shared
                .inbox
                .lock()
                .expect("inbox mutex")
                .send(Some(msg.clone()));
        }
    } else if let Some(id) = obj.get("id") {
        if id.is_array() || id.is_object() {
            log!("[jsonrpc] dropping response whose id cannot identify a request");
            return;
        }
        resolve(handle, obj);
    }
}

fn resolve(handle: &PeerHandle, msg: &Map<String, Value>) {
    let Some(id) = msg.get("id").and_then(Value::as_str) else {
        return;
    };
    let pending = handle
        .shared
        .pending
        .lock()
        .expect("pending mutex")
        .remove(id);
    let Some(pending) = pending else { return };

    // Everything from here must reach `settle`. The request has already been
    // taken off `pending`, so a failure in between does not merely drop a
    // message — it leaves the caller blocked until its timeout with nothing
    // left to answer it. A non-object `error` is enough to do that.
    let outcome = match msg.get("error") {
        Some(Value::Null) | None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
        Some(Value::Object(err)) => Err(RpcError {
            code: err
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or(INTERNAL_ERROR),
            message: err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string(),
            data: err.get("data").cloned(),
        }),
        Some(other) => Err(RpcError::new(
            INTERNAL_ERROR,
            format!("malformed error field: {other}"),
        )),
    };
    pending.settle(outcome);
}

fn dispatch(handle: &PeerHandle, handler: &dyn Handler, msg: &Value) {
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned();
    let id = msg.get("id").cloned();
    let is_request = id.is_some();

    // A handler bug must not kill the loop.
    let outcome = catch_unwind(AssertUnwindSafe(|| handler(method, params, is_request)));

    let error = match outcome {
        Ok(Ok(result)) => {
            if let Some(id) = id {
                let result = if result.is_null() { json!({}) } else { result };
                handle.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
            }
            return;
        }
        Ok(Err(err)) => err,
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "handler panicked".into());
            log!("[jsonrpc] unhandled panic in {method}: {detail}");
            RpcError::new(INTERNAL_ERROR, detail)
        }
    };

    match id {
        Some(id) => handle.send(&json!({"jsonrpc": "2.0", "id": id, "error": error.to_json()})),
        None => log!(
            "[jsonrpc] error in notification {method}: {}",
            error.message
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;
    use std::net::{Shutdown, TcpListener, TcpStream};

    /// A `Peer` wired to a fake counterpart over a real socket.
    ///
    /// A socket rather than an in-memory buffer because half of what these tests
    /// pin is what happens when the *pipe* dies, which a `Vec<u8>` cannot model.
    /// `std` has no portable `pipe()`, and a loopback socket closes the same way
    /// on Windows as on Linux.
    struct Pipes {
        peer: Peer,
        handle: PeerHandle,
        client_tx: TcpStream,
        client_rx: BufReader<TcpStream>,
    }

    impl Pipes {
        fn new() -> Self {
            Self::with_handler(|_, _, _| Ok(json!({})))
        }

        fn with_handler(handler: impl Handler + 'static) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            let client = TcpStream::connect(addr).expect("connect");
            let (server, _) = listener.accept().expect("accept");

            let rx = Box::new(BufReader::new(server.try_clone().expect("clone")));
            let mut peer = Peer::new(rx, Box::new(server));
            let handle = peer.handle();
            peer.start(handler);

            Pipes {
                peer,
                handle,
                client_rx: BufReader::new(client.try_clone().expect("clone")),
                client_tx: client,
            }
        }

        /// The next message the agent sent us.
        fn read_message(&mut self) -> Value {
            let mut line = String::new();
            self.client_rx.read_line(&mut line).expect("read");
            serde_json::from_str(&line).expect("valid json")
        }

        fn send_raw(&mut self, text: &str) {
            writeln!(self.client_tx, "{text}").expect("write");
            self.client_tx.flush().expect("flush");
        }

        fn reply(&mut self, id: &Value, result: Value) {
            let msg = json!({"jsonrpc": "2.0", "id": id, "result": result});
            self.send_raw(&serde_json::to_string(&msg).expect("serialise"));
        }

        fn hang_up(&self) {
            let _ = self.client_tx.shutdown(Shutdown::Both);
        }
    }

    impl Drop for Pipes {
        fn drop(&mut self) {
            self.handle.close();
            let _ = self.client_tx.shutdown(Shutdown::Both);
            self.peer.wait();
        }
    }

    /// Run `request` on its own thread, returning a join handle for the outcome.
    fn call_async(
        handle: &PeerHandle,
        timeout: Option<Duration>,
        abort: Option<Arc<AtomicBool>>,
    ) -> JoinHandle<Result<Value, RpcError>> {
        let handle = handle.clone();
        std::thread::spawn(move || {
            handle.request_with("probe", Some(json!({})), timeout, abort.as_ref())
        })
    }

    /// Give a spawned wait long enough to have ended if it was going to.
    fn settle() {
        std::thread::sleep(PeerHandle::WAIT_SLICE * 3);
    }

    // --------------------------------------------- surviving a bad message

    /// A single line from the editor used to end the reader permanently.
    ///
    /// In Python `{"method": []}` reached `method in FAST_PATH`, the unhashable
    /// key raised `TypeError`, and that escaped the loop's narrow `except`. The
    /// `finally` then closed the peer and stranded every pending request — so
    /// one bad line took the session's in-memory staged edits with it.
    #[test]
    fn one_malformed_message_does_not_kill_the_connection() {
        for bad in [
            r#"{"method": []}"#,
            r#"{"method": {"a": 1}}"#,
            r#"{"method": 7}"#,
            r#"{"id": [], "result": 1}"#,
            r#"{"id": {}, "result": 1}"#,
            r#"{"id": 1, "error": "not an object"}"#,
            "null",
            "[1, 2, 3]",
            "not json at all",
        ] {
            let mut pipes = Pipes::new();
            pipes.send_raw(bad);

            // The connection still works: a real request goes out and comes back.
            let call = call_async(&pipes.handle, Some(Duration::from_secs(5)), None);
            let sent = pipes.read_message();
            pipes.reply(&sent["id"], json!({"ok": true}));

            let got = call.join().expect("caller thread");
            assert_eq!(got, Ok(json!({"ok": true})), "the reader died on {bad}");
        }
    }

    /// `resolve` removes the pending request before it parses the error.
    ///
    /// A failure after that point is worse than a dropped message: nothing is
    /// left to answer the waiter, so it blocks until its own timeout with no
    /// error to show.
    #[test]
    fn a_malformed_error_field_still_releases_the_caller() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, Some(Duration::from_secs(10)), None);
        let sent = pipes.read_message();

        let msg = json!({"jsonrpc": "2.0", "id": sent["id"], "error": "a string, not an object"});
        pipes.send_raw(&serde_json::to_string(&msg).expect("serialise"));

        let got = call.join().expect("caller thread");
        assert!(got.is_err(), "the caller was left waiting");
    }

    /// A panicking handler is a bug in the agent, not grounds to drop the editor.
    #[test]
    fn a_panicking_handler_answers_with_an_error_and_keeps_serving() {
        let mut pipes = Pipes::with_handler(|method, _, _| {
            assert_ne!(method, "boom", "handler blew up");
            Ok(json!({"ok": true}))
        });

        pipes.send_raw(&json!({"jsonrpc": "2.0", "id": 1, "method": "boom"}).to_string());
        let reply = pipes.read_message();
        assert_eq!(reply["id"], json!(1));
        assert_eq!(reply["error"]["code"], json!(INTERNAL_ERROR));

        pipes.send_raw(&json!({"jsonrpc": "2.0", "id": 2, "method": "fine"}).to_string());
        let reply = pipes.read_message();
        assert_eq!(
            reply["result"],
            json!({"ok": true}),
            "the worker died with the handler"
        );
    }

    // ------------------------------------------------------- the happy path

    #[test]
    fn a_reply_is_returned_to_the_caller() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, Some(Duration::from_secs(5)), None);

        let sent = pipes.read_message();
        assert_eq!(sent["method"], json!("probe"));
        pipes.reply(&sent["id"], json!({"ok": true}));

        assert_eq!(call.join().expect("caller thread"), Ok(json!({"ok": true})));
    }

    /// A user reading a diff is not a timeout.
    #[test]
    fn an_untimed_wait_survives_a_slow_answer() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, None, None);
        let sent = pipes.read_message();

        settle();
        assert!(
            !call.is_finished(),
            "an untimed request must still be waiting"
        );

        pipes.reply(&sent["id"], json!({"outcome": "selected"}));
        assert_eq!(
            call.join().expect("caller thread"),
            Ok(json!({"outcome": "selected"}))
        );
    }

    // ----------------------------------------------------- how a wait ends

    /// The cancel path: nobody answers, and the turn is cancelled anyway.
    #[test]
    fn abort_releases_an_untimed_wait() {
        let mut pipes = Pipes::new();
        let abort = Arc::new(AtomicBool::new(false));
        let call = call_async(&pipes.handle, None, Some(Arc::clone(&abort)));
        pipes.read_message();

        abort.store(true, Ordering::SeqCst);
        let err = call
            .join()
            .expect("caller thread")
            .expect_err("must not succeed");
        assert!(err.message.contains("cancelled"), "{err}");
    }

    #[test]
    fn a_timeout_still_expires() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, Some(PeerHandle::WAIT_SLICE * 2), None);
        pipes.read_message();

        let err = call
            .join()
            .expect("caller thread")
            .expect_err("must not succeed");
        assert!(err.message.contains("timed out"), "{err}");
    }

    /// A departed editor must not strand the worker.
    #[test]
    fn a_closed_pipe_releases_an_untimed_wait() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, None, None);
        pipes.read_message();

        pipes.hang_up();
        assert!(
            call.join().expect("caller thread").is_err(),
            "a dead peer must end the wait"
        );
    }

    /// A late answer to an abandoned request must not be mistaken for a live one.
    #[test]
    fn an_aborted_request_stops_tracking_its_id() {
        let mut pipes = Pipes::new();
        let abort = Arc::new(AtomicBool::new(false));
        let call = call_async(&pipes.handle, None, Some(Arc::clone(&abort)));
        let sent = pipes.read_message();

        abort.store(true, Ordering::SeqCst);
        let _ = call.join().expect("caller thread");

        pipes.reply(&sent["id"], json!({"outcome": "selected"}));
        settle();

        assert_eq!(
            pipes.handle.pending_count(),
            0,
            "the abandoned request must be forgotten"
        );
    }

    // ------------------------------------------------------------ dispatch

    /// The reason the fast path exists: `session/cancel` arrives *during* the
    /// prompt it is meant to interrupt, so queueing it behind that prompt would
    /// deliver it only once the turn had already finished.
    #[test]
    fn a_fast_path_notification_overtakes_a_busy_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut client = TcpStream::connect(addr).expect("connect");
        let (server, _) = listener.accept().expect("accept");

        let release = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (h_release, h_cancelled) = (Arc::clone(&release), Arc::clone(&cancelled));

        let rx = Box::new(BufReader::new(server.try_clone().expect("clone")));
        let mut peer = Peer::new(rx, Box::new(server)).with_fast_path(|m| m == "cancel");
        let handle = peer.handle();
        peer.start(move |method: &str, _: Option<Value>, _: bool| {
            match method {
                "cancel" => h_cancelled.store(true, Ordering::SeqCst),
                // Occupies the worker exactly as a long turn would.
                _ => {
                    while !h_release.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            }
            Ok(json!({}))
        });

        writeln!(client, "{}", json!({"jsonrpc": "2.0", "method": "prompt"})).expect("write");
        writeln!(client, "{}", json!({"jsonrpc": "2.0", "method": "cancel"})).expect("write");
        client.flush().expect("flush");
        settle();

        assert!(
            cancelled.load(Ordering::SeqCst),
            "cancel was queued behind the turn it was meant to interrupt",
        );
        release.store(true, Ordering::SeqCst);
        handle.close();
        let _ = client.shutdown(Shutdown::Both);
        peer.wait();
    }

    /// Framing is the contract: one JSON value per line, no embedded newlines.
    #[test]
    fn a_multiline_payload_is_still_written_as_one_line() {
        let mut pipes = Pipes::new();
        pipes
            .handle
            .notify("log", Some(json!({"text": "two\nlines"})));

        let mut line = String::new();
        pipes.client_rx.read_line(&mut line).expect("read");
        let msg: Value = serde_json::from_str(&line).expect("one value per line");
        assert_eq!(msg["params"]["text"], json!("two\nlines"));
    }

    /// Writing after the peer closed is a no-op, not a panic.
    #[test]
    fn a_closed_peer_drops_writes_instead_of_failing() {
        let pipes = Pipes::new();
        pipes.handle.close();
        pipes
            .handle
            .notify("log", Some(json!({"text": "into the void"})));
        assert!(pipes.handle.is_closed());
    }

    /// Reading is not required for the port to be useful, but a silent
    /// truncation would be: a reply with no `result` key is a null result, not a
    /// missing one.
    #[test]
    fn a_reply_without_a_result_resolves_to_null() {
        let mut pipes = Pipes::new();
        let call = call_async(&pipes.handle, Some(Duration::from_secs(5)), None);
        let sent = pipes.read_message();

        pipes.send_raw(&json!({"jsonrpc": "2.0", "id": sent["id"]}).to_string());
        assert_eq!(call.join().expect("caller thread"), Ok(Value::Null));
    }

    /// The reader must not be blocked by an unread socket, which is what
    /// `read_to_end` on the client side would cause if writes were unbuffered.
    #[test]
    fn many_notifications_do_not_wedge_the_writer() {
        let mut pipes = Pipes::new();
        for i in 0..200 {
            pipes.handle.notify("tick", Some(json!({"n": i})));
        }
        let mut seen = 0;
        for _ in 0..200 {
            let msg = pipes.read_message();
            assert_eq!(msg["params"]["n"], json!(seen));
            seen += 1;
        }
        assert_eq!(seen, 200);
    }

    #[test]
    fn an_error_carries_its_data_through_the_wire() {
        let err = RpcError::with_data(INVALID_PARAMS, "bad", json!({"field": "path"}));
        assert_eq!(
            err.to_json(),
            json!({"code": INVALID_PARAMS, "message": "bad", "data": {"field": "path"}}),
        );
        assert_eq!(
            RpcError::new(METHOD_NOT_FOUND, "nope").to_json(),
            json!({"code": METHOD_NOT_FOUND, "message": "nope"}),
            "an absent data field must stay absent, not become null",
        );
    }

    /// Unused today, but the codes are the protocol's and a typo here would be
    /// invisible until an editor rejected a message.
    #[test]
    fn the_standard_error_codes_are_the_ones_the_spec_names() {
        assert_eq!(
            (
                PARSE_ERROR,
                INVALID_REQUEST,
                METHOD_NOT_FOUND,
                INVALID_PARAMS,
                INTERNAL_ERROR
            ),
            (-32700, -32600, -32601, -32602, -32603),
        );
    }
}
