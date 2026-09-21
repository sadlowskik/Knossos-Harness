//! Bounded windows over the log for the HTTP API. Port of `parseEventPage`
//! and `pageResult` in `field/server/src/api.js`.

use super::Event;
use serde::Serialize;

pub const EVENT_PAGE_DEFAULT: usize = 2_000;
pub const EVENT_PAGE_MAX: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventPage {
    pub from: u64,
    pub limit: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{message}")]
pub struct PaginationError {
    /// Stable code the API returns: always `invalid_pagination`.
    pub code: &'static str,
    pub message: String,
}

impl PaginationError {
    fn new(message: impl Into<String>) -> Self {
        PaginationError {
            code: "invalid_pagination",
            message: message.into(),
        }
    }
}

/// A JavaScript-style integer parse: the text must be a finite number with
/// no fractional part inside the safe-integer range, so `1.5`, `nope` and
/// `1e3` are rejected while `4` and `004` are accepted.
fn safe_integer(text: &str) -> Option<u64> {
    let text = text.trim();
    if text.is_empty() || text.contains(['e', 'E', 'x', 'X']) {
        return None;
    }
    let value: f64 = text.parse().ok()?;
    if !value.is_finite()
        || value.fract() != 0.0
        || !(0.0..=9_007_199_254_740_991.0).contains(&value)
    {
        return None;
    }
    Some(value as u64)
}

/// Parse the `from` and `limit` query parameters as the API receives them
/// (`None` when absent).
pub fn parse_event_page(
    from: Option<&str>,
    limit: Option<&str>,
) -> Result<EventPage, PaginationError> {
    let from = match from {
        None | Some("") => 0,
        Some(raw) => safe_integer(raw)
            .ok_or_else(|| PaginationError::new("from must be a non-negative integer"))?,
    };
    let limit = match limit {
        None | Some("") => EVENT_PAGE_DEFAULT,
        Some(raw) => match safe_integer(raw) {
            Some(n) if (1..=EVENT_PAGE_MAX as u64).contains(&n) => n as usize,
            _ => {
                return Err(PaginationError::new(format!(
                    "limit must be an integer between 1 and {EVENT_PAGE_MAX}"
                )))
            }
        },
    };
    Ok(EventPage { from, limit })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageResult {
    pub events: Vec<Event>,
    pub head: u64,
    pub next_from: Option<u64>,
}

/// Cut `events` to the page and compute the cursor. `has_more` defaults to
/// "the page was full", which is what an unbounded read tells us.
pub fn page_result(
    events: Vec<Event>,
    page: EventPage,
    head: Option<u64>,
    has_more: Option<bool>,
) -> PageResult {
    let has_more = has_more.unwrap_or(events.len() == page.limit);
    let rows: Vec<Event> = events.into_iter().take(page.limit).collect();
    let last = rows.last().map(|e| e.seq);
    PageResult {
        next_from: if has_more { last } else { None },
        head: head.unwrap_or(last.unwrap_or(page.from)),
        events: rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_explicit_values_parse() {
        assert_eq!(
            parse_event_page(None, None).unwrap(),
            EventPage {
                from: 0,
                limit: 2000
            }
        );
        assert_eq!(
            parse_event_page(Some(""), Some("")).unwrap(),
            EventPage {
                from: 0,
                limit: 2000
            }
        );
        assert_eq!(
            parse_event_page(Some("4"), Some("3")).unwrap(),
            EventPage { from: 4, limit: 3 }
        );
    }

    #[test]
    fn every_invalid_form_from_the_reference_suite_is_rejected() {
        for (from, limit) in [
            (Some("-1"), None),
            (Some("1.5"), None),
            (Some("nope"), None),
            (None, Some("0")),
            (None, Some("10001")),
            (None, Some("1.2")),
            (None, Some("nope")),
        ] {
            let err = parse_event_page(from, limit).unwrap_err();
            assert_eq!(err.code, "invalid_pagination", "{from:?} {limit:?}");
        }
    }
}
