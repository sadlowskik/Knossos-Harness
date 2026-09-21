//! Page through the append-only log without assuming an upper event count.
//! Port of `field/server/src/store/replay.js`.
//!
//! Boot, trace export and historical replay in the UI all walk the log; one
//! helper keeps them from drifting apart, and a `to` ceiling gives an exact
//! historical point, which is what makes "replay to seq N" a determinism
//! oracle for the projection.

use super::Event;

/// Anything that can serve events after a sequence number, oldest first.
pub trait ReadEvents {
    fn read(&self, from_seq: u64, limit: usize) -> Vec<Event>;
}

/// Anything that folds events into state.
pub trait ApplyEvent {
    fn apply(&mut self, event: &Event);
}

impl<F: FnMut(&Event)> ApplyEvent for F {
    fn apply(&mut self, event: &Event) {
        self(event)
    }
}

/// Which slice of the log to walk.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    /// Events with `seq > from` are returned.
    pub from: u64,
    /// Inclusive ceiling on `seq`; `None` walks to the end.
    pub to: Option<u64>,
    pub page_size: usize,
}

impl Default for Range {
    fn default() -> Self {
        Range {
            from: 0,
            to: None,
            page_size: 10_000,
        }
    }
}

impl Range {
    pub fn to(mut self, to: u64) -> Self {
        self.to = Some(to);
        self
    }

    pub fn page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size.max(1);
        self
    }
}

pub fn read_event_range(log: &impl ReadEvents, range: Range) -> Vec<Event> {
    let mut events = Vec::new();
    let mut cursor = range.from;
    let ceiling = range.to.map(|to| to.max(cursor));
    let page = range.page_size.max(1);

    while ceiling.is_none_or(|c| cursor < c) {
        let batch = log.read(cursor, page);
        let Some(last) = batch.last() else {
            break;
        };
        let next = last.seq;
        let short = batch.len() < page;
        for event in batch {
            if ceiling.is_some_and(|c| event.seq > c) {
                return events;
            }
            events.push(event);
        }
        if next <= cursor {
            break;
        }
        cursor = next;
        if short {
            break;
        }
    }
    events
}

/// What a replay produced.
#[derive(Debug, Clone)]
pub struct Replayed {
    pub count: usize,
    pub last_event: Option<Event>,
}

pub fn replay_into(
    log: &impl ReadEvents,
    projection: &mut impl ApplyEvent,
    range: Range,
) -> Replayed {
    let events = read_event_range(log, range);
    for event in &events {
        projection.apply(event);
    }
    Replayed {
        count: events.len(),
        last_event: events.last().cloned(),
    }
}
