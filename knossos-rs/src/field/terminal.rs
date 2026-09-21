//! The operator terminal: one shell command per run, streamed to every
//! browser as `terminal` frames. Port of `field/server/src/terminal-policy.js`
//! and the `/api/terminal/*` routes of `field/server/src/api.js`.
//!
//! Frames are exactly what the Node server sends:
//! `{ type: 'terminal', id, stream: 'out' | 'err' | 'meta', data, exit? }`.

use super::child_env::{build_child_environment, process_environment};
use super::hub::Hub;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

/// `TERMINAL_LIMITS`.
#[derive(Debug, Clone, Copy)]
pub struct TerminalLimits {
    pub concurrent: usize,
    pub timeout_ms: u64,
    pub output_bytes: usize,
    pub output_lines: usize,
    pub command_chars: usize,
}

pub const TERMINAL_LIMITS: TerminalLimits = TerminalLimits {
    concurrent: 4,
    timeout_ms: 10 * 60 * 1000,
    output_bytes: 2 * 1024 * 1024,
    output_lines: 20_000,
    command_chars: 16_000,
};

/// `validateTerminalCommand`: the message is what the API reports.
pub fn validate_terminal_command(value: Option<&Value>) -> Result<String, String> {
    let Some(text) = value
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty())
    else {
        return Err("terminal command is required".into());
    };
    if text.chars().count() > TERMINAL_LIMITS.command_chars {
        return Err("terminal command is too long".into());
    }
    if text.contains('\0') {
        return Err("terminal command contains a null byte".into());
    }
    Ok(text.to_string())
}

/// `redactCommand`: bearer tokens, `sk-` keys and `secret=value` pairs.
pub fn redact_command(value: &str) -> String {
    static BEARER: OnceLock<Regex> = OnceLock::new();
    static SK: OnceLock<Regex> = OnceLock::new();
    static PAIR: OnceLock<Regex> = OnceLock::new();
    let bearer = BEARER.get_or_init(|| {
        Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b").expect("static regex")
    });
    let sk = SK.get_or_init(|| Regex::new(r"\b(sk-[A-Za-z0-9_-]{8,})\b").expect("static regex"));
    let pair = PAIR.get_or_init(|| {
        Regex::new(r"(?i)\b(password|passwd|secret|token|api[_-]?key|private[_-]?key)=([^\s;&|]+)")
            .expect("static regex")
    });
    let step = bearer.replace_all(value, "Bearer [REDACTED]");
    let step = sk.replace_all(&step, "[REDACTED]");
    pair.replace_all(&step, "${1}=[REDACTED]").into_owned()
}

/// What `accept` returns: the text that fits, and whether the budget is gone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub text: String,
    pub exceeded: bool,
}

/// `OutputBudget`: bytes and lines a terminal may stream before it is
/// stopped. Once exceeded, nothing more is accepted.
#[derive(Debug)]
pub struct OutputBudget {
    max_bytes: usize,
    max_lines: usize,
    bytes: usize,
    lines: usize,
    exceeded: bool,
}

impl Default for OutputBudget {
    fn default() -> Self {
        OutputBudget::new(TERMINAL_LIMITS.output_bytes, TERMINAL_LIMITS.output_lines)
    }
}

impl OutputBudget {
    pub fn new(bytes: usize, lines: usize) -> Self {
        OutputBudget {
            max_bytes: bytes,
            max_lines: lines,
            bytes: 0,
            lines: 0,
            exceeded: false,
        }
    }

    pub fn accept(&mut self, chunk: &[u8]) -> Accepted {
        if self.exceeded {
            return Accepted {
                text: String::new(),
                exceeded: true,
            };
        }
        let remaining = self.max_bytes.saturating_sub(self.bytes);
        let accepted = &chunk[..chunk.len().min(remaining)];
        let mut text = String::from_utf8_lossy(accepted).into_owned();
        let remaining_lines = self.max_lines.saturating_sub(self.lines);
        let breaks: Vec<usize> = text.match_indices('\n').map(|(i, _)| i).collect();
        if breaks.len() > remaining_lines {
            text = if remaining_lines == 0 {
                String::new()
            } else {
                text[..breaks[remaining_lines - 1] + 1].to_string()
            };
        }
        self.bytes += text.len();
        self.lines += breaks.len().min(remaining_lines);
        self.exceeded = chunk.len() > remaining || breaks.len() > remaining_lines;
        Accepted {
            text,
            exceeded: self.exceeded,
        }
    }
}

/// A running shell and the handle that stops it.
#[derive(Debug)]
pub struct TerminalProcess {
    pid: u32,
    child: Mutex<Child>,
}

impl TerminalProcess {
    pub fn pid(&self) -> u32 {
        self.pid
    }
}

/// `stopProcessTree`: the whole tree on Windows via `taskkill`, the child's
/// own process group elsewhere, then the child itself as the fallback.
pub fn stop_process_tree(process: &TerminalProcess) {
    let pid = process.pid;
    if pid == 0 {
        return;
    }
    #[cfg(windows)]
    {
        let mut killer = Command::new("taskkill.exe");
        killer
            .args(["/pid", &pid.to_string(), "/t", "/f"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        hide_window(&mut killer);
        if killer.spawn().is_err() {
            if let Ok(mut child) = process.child.lock() {
                let _ = child.kill();
            }
        }
    }
    #[cfg(unix)]
    {
        // The child leads its own process group (`process_group(0)` at
        // spawn), so its negative pid names exactly that group.
        let group = -(pid as i32);
        if group < -1 && unix_kill(group, SIGTERM) != 0 {
            let _ = unix_kill(pid as i32, SIGTERM);
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        if let Ok(mut child) = process.child.lock() {
            let _ = child.kill();
        }
    }
}

#[cfg(unix)]
const SIGTERM: i32 = 15;

#[cfg(unix)]
fn unix_kill(pid: i32, signal: i32) -> i32 {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    // A guard against ever signalling "every process" (0 or -1).
    if pid == 0 || pid == -1 {
        return -1;
    }
    // SAFETY: `kill(2)` takes two plain integers and touches no memory.
    unsafe { kill(pid, signal) }
}

#[cfg(windows)]
fn hide_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_window(_command: &mut Command) {}

/// One active terminal: `terminals.get(id)` in the Node server.
#[derive(Debug)]
pub struct ActiveTerminal {
    pub process: TerminalProcess,
    timer: Mutex<Option<mpsc::Sender<()>>>,
    stopped_for_limit: AtomicBool,
}

impl ActiveTerminal {
    /// `clearTimeout(timer)`: the timeout thread wakes and exits.
    fn clear_timer(&self) {
        if let Ok(mut timer) = self.timer.lock() {
            timer.take();
        }
    }
}

/// The active-terminal map, shared by the API and the reader threads.
#[derive(Debug, Default)]
pub struct Terminals {
    active: Mutex<HashMap<String, Arc<ActiveTerminal>>>,
}

impl Terminals {
    pub fn new() -> Arc<Terminals> {
        Arc::new(Terminals::default())
    }

    pub fn len(&self) -> usize {
        self.active.lock().map(|a| a.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: &str) -> bool {
        self.active
            .lock()
            .map(|a| a.contains_key(id))
            .unwrap_or(false)
    }

    pub fn get(&self, id: &str) -> Option<Arc<ActiveTerminal>> {
        self.active.lock().ok().and_then(|a| a.get(id).cloned())
    }

    fn insert(&self, id: &str, terminal: Arc<ActiveTerminal>) {
        if let Ok(mut a) = self.active.lock() {
            a.insert(id.to_string(), terminal);
        }
    }

    fn remove(&self, id: &str) {
        if let Ok(mut a) = self.active.lock() {
            a.remove(id);
        }
    }

    /// `POST /api/terminal/kill`: true when the id was active.
    pub fn kill(&self, id: &str) -> bool {
        match self.get(id) {
            Some(terminal) => {
                terminal.clear_timer();
                stop_process_tree(&terminal.process);
                true
            }
            None => false,
        }
    }
}

/// What `POST /api/terminal/run` hands the runner once the workspace path,
/// the capacity and the command have been checked.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub id: String,
    pub workspace_id: String,
    /// `body.cwd || '.'`, as shown in the header frame.
    pub cwd_label: String,
    pub cwd: std::path::PathBuf,
    pub command: String,
}

/// One stream's reader: forwards chunks within the budget, stops the tree
/// the first time the budget is exceeded.
struct Forward {
    hub: Arc<Hub>,
    terminal: Arc<ActiveTerminal>,
    budget: Arc<Mutex<OutputBudget>>,
    id: String,
    stream: &'static str,
}

fn forward(f: Forward, source: Option<Box<dyn Read + Send>>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let Some(mut source) = source else {
            return;
        };
        let mut buffer = [0u8; 8192];
        loop {
            let n = match source.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            let accepted = match f.budget.lock() {
                Ok(mut b) => b.accept(&buffer[..n]),
                Err(_) => break,
            };
            if !accepted.text.is_empty() {
                f.hub.broadcast(&frame(&f.id, f.stream, accepted.text));
            }
            if accepted.exceeded && !f.terminal.stopped_for_limit.swap(true, Ordering::SeqCst) {
                f.hub.broadcast(&frame(
                    &f.id,
                    "meta",
                    "\n[terminated: output limit]\n".into(),
                ));
                stop_process_tree(&f.terminal.process);
            }
        }
    })
}

fn frame(id: &str, stream: &str, data: String) -> Value {
    json!({ "type": "terminal", "id": id, "stream": stream, "data": data })
}

fn shell_command(command: &str) -> Command {
    let mut cmd = if cfg!(windows) {
        let mut cmd = Command::new("powershell.exe");
        cmd.args(["-NoProfile", "-NonInteractive", "-Command", command]);
        cmd
    } else {
        let mut cmd = Command::new("bash");
        cmd.args(["-lc", command]);
        cmd
    };
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    hide_window(&mut cmd);
    cmd
}

/// Spawn the shell and stream it. The reply (`{ terminalId }`) is the
/// caller's; a shell that fails to start is reported as a frame, as in
/// Node where the `error` event fires after the route has answered.
pub fn run(terminals: &Arc<Terminals>, hub: &Arc<Hub>, request: RunRequest) {
    let RunRequest {
        id,
        workspace_id,
        cwd_label,
        cwd,
        command,
    } = request;
    let header = frame(
        &id,
        "meta",
        format!(
            "[{workspace_id}:{cwd_label}] $ {}\n",
            redact_command(&command)
        ),
    );
    let env = build_child_environment(&process_environment(), None, &[], &BTreeMap::new());
    let mut shell = shell_command(&command);
    shell
        .current_dir(&cwd)
        .env_clear()
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match shell.spawn() {
        Ok(child) => child,
        Err(e) => {
            hub.broadcast(&header);
            let mut failed = frame(&id, "err", format!("failed to start: {e}\n"));
            failed["exit"] = json!(-1);
            hub.broadcast(&failed);
            return;
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (timer_tx, timer_rx) = mpsc::channel::<()>();
    let terminal = Arc::new(ActiveTerminal {
        process: TerminalProcess {
            pid: child.id(),
            child: Mutex::new(child),
        },
        timer: Mutex::new(Some(timer_tx)),
        stopped_for_limit: AtomicBool::new(false),
    });
    terminals.insert(&id, Arc::clone(&terminal));

    // The time limit: a thread parked on the channel until the timeout, or
    // until `clear_timer` drops the sender.
    {
        let hub = Arc::clone(hub);
        let terminal = Arc::clone(&terminal);
        let id = id.clone();
        thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) =
                timer_rx.recv_timeout(Duration::from_millis(TERMINAL_LIMITS.timeout_ms))
            {
                terminal.stopped_for_limit.store(true, Ordering::SeqCst);
                hub.broadcast(&frame(&id, "meta", "\n[terminated: time limit]\n".into()));
                stop_process_tree(&terminal.process);
            }
        });
    }

    hub.broadcast(&header);
    let budget = Arc::new(Mutex::new(OutputBudget::default()));
    let out = forward(
        Forward {
            hub: Arc::clone(hub),
            terminal: Arc::clone(&terminal),
            budget: Arc::clone(&budget),
            id: id.clone(),
            stream: "out",
        },
        stdout.map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    let err = forward(
        Forward {
            hub: Arc::clone(hub),
            terminal: Arc::clone(&terminal),
            budget,
            id: id.clone(),
            stream: "err",
        },
        stderr.map(|s| Box::new(s) as Box<dyn Read + Send>),
    );

    // `close`: both streams drained and the process reaped.
    let terminals = Arc::clone(terminals);
    let hub = Arc::clone(hub);
    thread::spawn(move || {
        let _ = out.join();
        let _ = err.join();
        let status = terminal
            .process
            .child
            .lock()
            .ok()
            .and_then(|mut c| c.wait().ok());
        terminal.clear_timer();
        terminals.remove(&id);
        let code = status.and_then(|s| s.code());
        let mut exit = frame(
            &id,
            "meta",
            format!(
                "\n[exit {}]\n",
                code.map(|c| c.to_string()).unwrap_or_else(|| "null".into())
            ),
        );
        exit["exit"] = json!(code);
        hub.broadcast(&exit);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `terminal-policy.test.mjs`.
    #[test]
    fn validation_redaction_and_budgets() {
        assert_eq!(
            validate_terminal_command(Some(&json!("npm test"))).unwrap(),
            "npm test"
        );
        assert!(validate_terminal_command(Some(&json!("")))
            .unwrap_err()
            .contains("required"));
        assert!(validate_terminal_command(None)
            .unwrap_err()
            .contains("required"));
        assert!(validate_terminal_command(Some(&json!(42)))
            .unwrap_err()
            .contains("required"));
        assert!(validate_terminal_command(Some(&json!("bad\0command")))
            .unwrap_err()
            .contains("null byte"));
        let long = "x".repeat(TERMINAL_LIMITS.command_chars + 1);
        assert!(validate_terminal_command(Some(&json!(long)))
            .unwrap_err()
            .contains("too long"));

        let redacted = redact_command(
            "curl -H \"Authorization: Bearer abcdefghijklmnop\" x; TOKEN=secretvalue API_KEY=anothersecret sk-testcredential",
        );
        assert!(!redacted.contains("abcdefghijklmnop"), "{redacted}");
        assert!(!redacted.contains("secretvalue"), "{redacted}");
        assert!(!redacted.contains("anothersecret"), "{redacted}");
        assert!(!redacted.contains("sk-testcredential"), "{redacted}");
        assert_eq!(redact_command("echo hi"), "echo hi");
        assert_eq!(redact_command("TOKEN=abc ls"), "TOKEN=[REDACTED] ls");

        let mut bytes = OutputBudget::new(5, 10);
        assert_eq!(
            bytes.accept(b"1234"),
            Accepted {
                text: "1234".into(),
                exceeded: false
            }
        );
        assert_eq!(
            bytes.accept(b"567"),
            Accepted {
                text: "5".into(),
                exceeded: true
            }
        );
        assert_eq!(
            bytes.accept(b"ignored"),
            Accepted {
                text: String::new(),
                exceeded: true
            }
        );

        let mut lines = OutputBudget::new(100, 2);
        assert_eq!(
            lines.accept(b"one\ntwo\nthree\n"),
            Accepted {
                text: "one\ntwo\n".into(),
                exceeded: true
            }
        );
        let mut exact = OutputBudget::new(100, 2);
        assert_eq!(
            exact.accept(b"a\nb\n"),
            Accepted {
                text: "a\nb\n".into(),
                exceeded: false
            }
        );
        assert_eq!(
            exact.accept(b"c\n"),
            Accepted {
                text: String::new(),
                exceeded: true
            }
        );
    }

    #[test]
    fn frames_carry_the_node_shape() {
        let f = frame("t1", "out", "hi\n".into());
        assert_eq!(
            f,
            json!({ "type": "terminal", "id": "t1", "stream": "out", "data": "hi\n" })
        );
        let terminals = Terminals::new();
        assert!(terminals.is_empty());
        assert!(!terminals.kill("missing"));
        assert!(!terminals.contains("missing"));
    }
}
