//! Replayable dollar reservations. Port of `field/server/src/budget-ledger.js`.
//!
//! Missing cost is never treated as zero cost. Unsettled interrupted work
//! retains its reservation because its final bill is unknown.

use super::eventlog::{Event, Source};
use super::js::{finite, get_str, is_truthy, jnum, js_string, OrderedMap};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Reservation {
    pub session_id: String,
    pub campaign_id: Option<String>,
    pub limit_usd: f64,
    pub spent_usd: Option<f64>,
    pub terminal: bool,
    pub settled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignBudget {
    pub spent_usd: f64,
    pub reserved_usd: f64,
    pub remaining_usd: f64,
    pub unknown_cost_sessions: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BudgetLedger {
    reservations: OrderedMap<Reservation>,
}

fn session_key(data: &Value) -> String {
    data.get("sessionId")
        .map(js_string)
        .unwrap_or_else(|| "undefined".to_string())
}

impl BudgetLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, event: &Event) {
        let d = &event.data;
        if event.source == Source::Synthetic || is_truthy(d, "simulated") {
            return;
        }
        let key = session_key(d);
        if event.kind == "budget.reserved" {
            if let Some(limit) = finite(d, "limitUsd").filter(|l| *l > 0.0) {
                if !self.reservations.contains(&key) {
                    self.reservations.insert(
                        key.clone(),
                        Reservation {
                            session_id: key,
                            campaign_id: get_str(d, "campaignId").map(str::to_string),
                            limit_usd: limit,
                            spent_usd: None,
                            terminal: false,
                            settled: false,
                        },
                    );
                }
            }
            return;
        }
        let Some(entry) = self.reservations.get_mut(&key) else {
            return;
        };
        let kind = event.kind.as_str();
        let state = get_str(d, "state");
        if kind == "budget.reactivated" {
            entry.terminal = false;
            entry.settled = false;
        }
        if kind == "session.usage" {
            if let Some(cost) = finite(d, "costUsd").filter(|c| *c >= 0.0) {
                entry.spent_usd = Some(entry.spent_usd.unwrap_or(0.0).max(cost));
            }
        }
        if kind == "session.turn_complete" {
            entry.settled = true;
        }
        if kind == "session.state" && matches!(state, Some("spawning" | "thinking" | "working")) {
            entry.settled = false;
            entry.terminal = false;
        }
        if kind == "session.ended" || (kind == "session.state" && state == Some("interrupted")) {
            entry.terminal = true;
        }
    }

    pub fn reserved(entry: &Reservation) -> f64 {
        if entry.terminal && entry.settled && entry.spent_usd.is_some() {
            return 0.0;
        }
        (entry.limit_usd - entry.spent_usd.unwrap_or(0.0)).max(0.0)
    }

    pub fn campaign(&self, campaign_id: &str, spent_usd: f64, limit_usd: f64) -> CampaignBudget {
        let entries: Vec<&Reservation> = self
            .reservations
            .values()
            .filter(|e| e.campaign_id.as_deref() == Some(campaign_id))
            .collect();
        let reserved_usd: f64 = entries.iter().map(|e| Self::reserved(e)).sum();
        CampaignBudget {
            spent_usd,
            reserved_usd,
            remaining_usd: (limit_usd - spent_usd - reserved_usd).max(0.0),
            unknown_cost_sessions: entries
                .iter()
                .filter(|e| e.spent_usd.is_none() || (e.terminal && !e.settled))
                .count(),
        }
    }

    pub fn snapshot(&self) -> Vec<Value> {
        self.reservations
            .values()
            .map(|entry| {
                json!({
                    "sessionId": entry.session_id,
                    "campaignId": entry.campaign_id,
                    "limitUsd": jnum(entry.limit_usd),
                    "spentUsd": entry.spent_usd.map(jnum).unwrap_or(Value::Null),
                    "terminal": entry.terminal,
                    "settled": entry.settled,
                    "reservedUsd": jnum(Self::reserved(entry)),
                    "costStatus": match entry.spent_usd {
                        None => "missing",
                        Some(s) => {
                            if s == 0.0 {
                                "reported_zero"
                            } else {
                                "reported"
                            }
                        }
                    },
                    "finalCostKnown": entry.terminal && entry.settled && entry.spent_usd.is_some(),
                })
            })
            .collect()
    }
}
