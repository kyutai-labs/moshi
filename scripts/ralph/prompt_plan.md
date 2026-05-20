You are running one Ralph plan gate iteration.

Inputs:
- EPIC_ID={{EPIC_ID}}
- PLAN_REVIEW={{PLAN_REVIEW}}
- REQUIRE_PLAN_REVIEW={{REQUIRE_PLAN_REVIEW}}

Steps:
1) Re-anchor:
   - scripts/ralph/flowctl show {{EPIC_ID}} --json
   - scripts/ralph/flowctl cat {{EPIC_ID}}
   - git status
   - git log -10 --oneline

2) Save checkpoint (recovery point if context compacts during review cycles):
   ```bash
   scripts/ralph/flowctl checkpoint save --epic {{EPIC_ID}} --json
   ```

Ralph mode rules (must follow):
- If PLAN_REVIEW=rp: use `flowctl rp` wrappers (setup-review, select-add, prompt-get, chat-send).
- If PLAN_REVIEW=codex: use `flowctl codex` wrappers (plan-review with --receipt).
- Write receipt via bash heredoc (no Write tool) if `REVIEW_RECEIPT_PATH` set.
- If any rule is violated, output `<promise>RETRY</promise>` and stop.

3) Plan review gate:
   - If PLAN_REVIEW=rp: run `/flow-next:plan-review {{EPIC_ID}} --review=rp`
   - If PLAN_REVIEW=codex: run `/flow-next:plan-review {{EPIC_ID}} --review=codex`
   - If PLAN_REVIEW=export: run `/flow-next:plan-review {{EPIC_ID}} --review=export`
   - If PLAN_REVIEW=none:
     - If REQUIRE_PLAN_REVIEW=1: output `<promise>RETRY</promise>` and stop.
     - Else: set ship and stop:
       `scripts/ralph/flowctl epic set-plan-review-status {{EPIC_ID}} --status ship --json`

4) The skill will loop internally until `<verdict>SHIP</verdict>`:
   - First review uses `--new-chat`
   - If NEEDS_WORK: skill fixes plan AND syncs affected task specs, re-reviews in SAME chat (no --new-chat)
   - Repeats until SHIP
   - Only returns to Ralph after SHIP or MAJOR_RETHINK
   - If context compacts mid-review: `scripts/ralph/flowctl checkpoint restore --epic {{EPIC_ID}} --json`

5) IMMEDIATELY after SHIP verdict, write receipt (for rp mode):
   ```bash
   mkdir -p "$(dirname '{{REVIEW_RECEIPT_PATH}}')"
   ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
   cat > '{{REVIEW_RECEIPT_PATH}}' <<EOF
   {"type":"plan_review","id":"{{EPIC_ID}}","mode":"rp","timestamp":"$ts","iteration":{{RALPH_ITERATION}}}
   EOF
   ```
   For codex mode, receipt is written automatically by `flowctl codex plan-review --receipt`.
   **CRITICAL: Copy EXACTLY. The `"id":"{{EPIC_ID}}"` field is REQUIRED.**
   Missing id = verification fails = forced retry.

6) After SHIP:
   - `scripts/ralph/flowctl epic set-plan-review-status {{EPIC_ID}} --status ship --json`
   - stop (do NOT output promise tag)

7) If MAJOR_RETHINK (rare):
   - `scripts/ralph/flowctl epic set-plan-review-status {{EPIC_ID}} --status needs_work --json`
   - output `<promise>FAIL</promise>` and stop

8) On hard failure, output `<promise>FAIL</promise>` and stop.

## ⛔ FORBIDDEN OUTPUT
**NEVER output `<promise>COMPLETE</promise>`** — this prompt handles ONE epic only.
Ralph detects all-work-complete automatically via the selector. Outputting COMPLETE here is INVALID and will be ignored.
