# Admission and restart behavior

Field reserves a session's dollar allocation in the event log before starting its
harness. A campaign's remaining allocation is its limit minus observed spend and
outstanding reservations. A requested allocation is narrowed to that remainder;
no remainder means no spawn. Duplicate or older cumulative usage cannot refund
spend or inflate totals. Nonfinite and negative usage is ignored.

An ended session releases its remaining allocation only after a completed turn
and an explicit cost report. Missing final billing retains the reservation across
replay. Reported zero is distinct from missing telemetry. This conservative policy
can block a campaign after an unbilled failure: explicit bill reconciliation is
still required production work. Never edit event history to manufacture a refund.
Pausing at reported usage boundaries cannot guarantee a provider's final charge
stays below a dollar ceiling; provider-side ceilings remain an acceptance gate.

Global live harness capacity defaults to 16, configured with
`defaults.max_concurrent_sessions` in `field.yaml`. Each endpoint defaults to 4,
overridden by its `max_concurrent_sessions`. Campaign `concurrency` is enforced
at registry admission as well as by the director. Limits must be positive integers.
Spawn, resume, assignment, escalation and endpoint failover check admission.
Draining children count until their close event. Old process callbacks cannot
terminate a replacement session. These are session caps, not OS process-tree or
GPU memory limits. Capacity denial emits `capacity.denied`; a durable fair queue
and complete admitted/released event lifecycle remain open.

Routines support UTC schedules and skip-overlap behavior. Field persists a minute
claim before spawning; a restart in that minute cannot duplicate the run, and
backward clock steps cannot replay old slots. A crash after a claim but before
spawn may skip that run. Missed downtime runs are not backfilled. An unfinished
run is marked failed on restart. The UI shows the next UTC run, owner and recent
history. Other timezones and overlap policies are rejected rather than ignored.

Browser bootstrap links expire after ten minutes. Sign out clears and revokes the
browser session, closes existing WebSockets and stops reconnect attempts. Running
harness capabilities are independent. A fresh server bootstrap is required to
sign in again; CLI logout/re-enrollment remains unfinished.
