You are running one Ralph work iteration.

Inputs:
- TASK_ID={{TASK_ID}}
- BRANCH_MODE={{BRANCH_MODE_EFFECTIVE}}
- WORK_REVIEW={{WORK_REVIEW}}

## Steps (execute ALL in order)

**Step 1: Execute task**
```
/flow-next:work {{TASK_ID}} --branch={{BRANCH_MODE_EFFECTIVE}} --review={{WORK_REVIEW}}
```
When `--review=rp`, the worker subagent invokes `/flow-next:impl-review` internally.
When `--review=codex`, the worker uses `flowctl codex impl-review` for review.
The impl-review skill handles review coordination and requires `<verdict>SHIP|NEEDS_WORK|MAJOR_RETHINK</verdict>` from reviewer.
Do NOT improvise review prompts - the skill has the correct format.

**Step 2: Verify task done** (AFTER skill returns)
```bash
scripts/ralph/flowctl show {{TASK_ID}} --json
```
If status != `done`, output `<promise>RETRY</promise>` and stop.

**Step 3: Write impl receipt** (MANDATORY if WORK_REVIEW=rp or codex)
For rp mode:
```bash
mkdir -p "$(dirname '{{REVIEW_RECEIPT_PATH}}')"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > '{{REVIEW_RECEIPT_PATH}}' <<EOF
{"type":"impl_review","id":"{{TASK_ID}}","mode":"rp","timestamp":"$ts","iteration":{{RALPH_ITERATION}}}
EOF
echo "Receipt written: {{REVIEW_RECEIPT_PATH}}"
```
For codex mode, receipt is written automatically by `flowctl codex impl-review --receipt`.
**CRITICAL: Copy the command EXACTLY. The `"id":"{{TASK_ID}}"` field is REQUIRED.**
Ralph verifies receipts match this exact schema. Missing id = verification fails = forced retry.

**Step 4: Validate epic**
```bash
scripts/ralph/flowctl validate --epic $(echo {{TASK_ID}} | sed 's/\.[0-9]*$//') --json
```

**Step 5: On hard failure** → output `<promise>FAIL</promise>` and stop.

## Rules
- Must run `flowctl done` and verify task status is `done` before commit.
- Must `git add -A` (never list files).
- Do NOT use TodoWrite.

## ⛔ FORBIDDEN OUTPUT
**NEVER output `<promise>COMPLETE</promise>`** — this prompt handles ONE task only.
Ralph detects all-work-complete automatically via the selector. Outputting COMPLETE here is INVALID and will be ignored.
