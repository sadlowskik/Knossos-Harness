//! Per-agent context budgeting.
//!
//! A model's advertised context is not a prompt budget: output tokens, tool
//! protocol framing, and a safety margin all occupy the same window. This
//! module resolves those constraints once, immediately before a turn.

pub const DEFAULT_SAFETY_PERCENT: u8 = 80;
pub const DEFAULT_PROTOCOL_RESERVE: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContextPolicy {
    pub assigned_tokens: Option<u32>,
    pub compact_at_tokens: Option<u32>,
    pub safety_percent: u8,
    pub completion_reserve: u32,
    pub protocol_reserve: u32,
    pub compaction_enabled: bool,
}

impl ContextPolicy {
    pub fn for_completion(completion_reserve: u32) -> Self {
        Self {
            assigned_tokens: None,
            compact_at_tokens: None,
            safety_percent: DEFAULT_SAFETY_PERCENT,
            completion_reserve,
            protocol_reserve: DEFAULT_PROTOCOL_RESERVE,
            compaction_enabled: true,
        }
    }

    /// Resolve a requested allocation against the server window. An explicit
    /// assignment can size an unknown backend, but never raise a known limit.
    pub fn resolve(self, engine_tokens: Option<u32>, fallback_prompt: u32) -> ContextBudget {
        let safety = self.safety_percent.clamp(1, 100) as u32;
        let safe_engine = engine_tokens.map(|tokens| tokens.saturating_mul(safety) / 100);
        let assigned = match (self.assigned_tokens, safe_engine) {
            (Some(wanted), Some(limit)) => wanted.min(limit),
            (Some(wanted), None) => wanted,
            (None, Some(limit)) => limit,
            (None, None) => fallback_prompt
                .saturating_add(self.completion_reserve)
                .saturating_add(self.protocol_reserve),
        }
        .max(1);
        let protocol_reserve = self.protocol_reserve.min(assigned.saturating_sub(1));
        let completion_reserve = self
            .completion_reserve
            .min((assigned / 4).max(1))
            .min(assigned.saturating_sub(protocol_reserve).saturating_sub(1));
        let input_limit = assigned
            .saturating_sub(completion_reserve)
            .saturating_sub(protocol_reserve)
            .max(1);
        let default_trigger = input_limit.saturating_mul(80) / 100;
        let compact_at = self
            .compact_at_tokens
            .unwrap_or(default_trigger)
            .clamp(1, input_limit);
        ContextBudget {
            engine_tokens,
            assigned_tokens: assigned,
            input_limit_tokens: input_limit,
            compact_at_tokens: compact_at,
            completion_reserve,
            protocol_reserve,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub engine_tokens: Option<u32>,
    pub assigned_tokens: u32,
    pub input_limit_tokens: u32,
    pub compact_at_tokens: u32,
    pub completion_reserve: u32,
    pub protocol_reserve: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_safe_window_and_reserves_output() {
        let budget = ContextPolicy::for_completion(4_096).resolve(Some(32_768), 24_000);
        assert_eq!(budget.assigned_tokens, 26_214);
        assert_eq!(budget.input_limit_tokens, 21_606);
        assert_eq!(budget.compact_at_tokens, 17_284);
    }

    #[test]
    fn explicit_agent_window_cannot_exceed_server_safety_ceiling() {
        let mut policy = ContextPolicy::for_completion(1_000);
        policy.assigned_tokens = Some(30_000);
        assert_eq!(policy.resolve(Some(20_000), 24_000).assigned_tokens, 16_000);
    }

    #[test]
    fn compact_override_is_clamped_to_real_input_room() {
        let mut policy = ContextPolicy::for_completion(2_000);
        policy.protocol_reserve = 500;
        policy.assigned_tokens = Some(8_000);
        policy.compact_at_tokens = Some(99_000);
        let budget = policy.resolve(None, 24_000);
        assert_eq!(budget.input_limit_tokens, 5_500);
        assert_eq!(budget.compact_at_tokens, 5_500);
    }

    #[test]
    fn a_small_agent_window_still_has_prompt_room() {
        let mut policy = ContextPolicy::for_completion(8_192);
        policy.assigned_tokens = Some(8_000);
        let budget = policy.resolve(None, 24_000);
        assert_eq!(budget.completion_reserve, 2_000);
        assert_eq!(budget.input_limit_tokens, 5_488);
    }
}
