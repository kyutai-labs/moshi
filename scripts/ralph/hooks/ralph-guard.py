#!/usr/bin/env python3
"""
Ralph Guard - Hook script for enforcing Ralph workflow rules.

Only runs when FLOW_RALPH=1 is set. Exits silently otherwise to avoid
polluting context for non-Ralph users.

Enforces:
- No --json flag on chat-send (suppresses review text)
- No --new-chat on re-reviews (loses reviewer context)
- Receipt must be written after SHIP verdict
- Validates flowctl command patterns

Supports both review backends:
- rp (RepoPrompt): tracks chat-send calls and receipt writes
- codex: tracks flowctl codex impl-review/plan-review and verdict output
"""

# Version for drift detection (bump when making changes)
RALPH_GUARD_VERSION = "0.12.0"

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def get_state_file(session_id: str) -> Path:
    """Get state file path for this session."""
    return Path(f"/tmp/ralph-guard-{session_id}.json")


def load_state(session_id: str) -> dict:
    """Load session state."""
    state_file = get_state_file(session_id)
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(), object_hook=state_decoder)
            # Ensure all expected keys exist
            state.setdefault("chats_sent", 0)
            state.setdefault("last_verdict", None)
            state.setdefault("window", None)
            state.setdefault("tab", None)
            state.setdefault("chat_send_succeeded", False)
            state.setdefault("flowctl_done_called", set())
            state.setdefault("codex_review_succeeded", False)
            return state
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return {
        "chats_sent": 0,
        "last_verdict": None,
        "window": None,
        "tab": None,
        "chat_send_succeeded": False,  # Track if chat-send actually returned review text
        "flowctl_done_called": set(),  # Track tasks that had flowctl done called
        "codex_review_succeeded": False,  # Track if codex review returned verdict
    }


def state_decoder(obj):
    """JSON decoder that handles sets."""
    if "flowctl_done_called" in obj and isinstance(obj["flowctl_done_called"], list):
        obj["flowctl_done_called"] = set(obj["flowctl_done_called"])
    return obj


def state_encoder(obj):
    """JSON encoder that handles sets."""
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_state(session_id: str, state: dict) -> None:
    """Save session state."""
    state_file = get_state_file(session_id)
    state_file.write_text(json.dumps(state, default=state_encoder))


def output_block(reason: str) -> None:
    """Output blocking response (exit code 2 style via stderr)."""
    print(reason, file=sys.stderr)
    sys.exit(2)


# --- Memory helpers ---


def get_repo_root() -> Path:
    """Find git repo root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return Path.cwd()


def is_memory_enabled() -> bool:
    """Check if memory is enabled in .flow/config.json."""
    config_path = get_repo_root() / ".flow" / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
        return config.get("memory", {}).get("enabled", False)
    except (json.JSONDecodeError, Exception):
        return False


def output_json(data: dict) -> None:
    """Output JSON response."""
    print(json.dumps(data))
    sys.exit(0)


def handle_pre_tool_use(data: dict) -> None:
    """Handle PreToolUse event - validate commands before execution."""
    tool_input = data.get("tool_input", {})
    command = tool_input.get("command", "")
    session_id = data.get("session_id", "unknown")

    # Check for chat-send commands
    if "chat-send" in command:
        # Block --json flag
        if re.search(r"chat-send.*--json", command):
            output_block(
                "BLOCKED: Do not use --json with chat-send. "
                "It suppresses the review text. Remove --json flag."
            )

        # Check for --new-chat on re-reviews
        if "--new-chat" in command:
            state = load_state(session_id)
            if state["chats_sent"] > 0:
                output_block(
                    "BLOCKED: Do not use --new-chat for re-reviews. "
                    "Stay in the same chat so reviewer has context. "
                    "Remove --new-chat flag."
                )

    # Block direct codex calls (must use flowctl codex wrappers)
    if re.search(r"\bcodex\b", command):
        # Allow flowctl codex wrappers
        is_wrapper = re.search(r"flowctl\s+codex|FLOWCTL.*codex", command)
        if not is_wrapper:
            # Block direct codex usage
            if re.search(r"\bcodex\s+exec\b", command):
                output_block(
                    "BLOCKED: Do not call 'codex exec' directly. "
                    "Use 'flowctl codex impl-review' or 'flowctl codex plan-review' "
                    "to ensure proper receipt handling and session continuity."
                )
            if re.search(r"\bcodex\s+review\b", command):
                output_block(
                    "BLOCKED: Do not call 'codex review' directly. "
                    "Use 'flowctl codex impl-review' or 'flowctl codex plan-review'."
                )
        # Block --last even through wrappers (breaks session continuity)
        if re.search(r"--last\b", command):
            output_block(
                "BLOCKED: Do not use '--last' with codex. "
                "Session continuity is managed via session_id in receipts."
            )

    # Validate setup-review usage
    if "setup-review" in command:
        if not re.search(r"--repo-root", command):
            output_block(
                "BLOCKED: setup-review requires --repo-root flag. "
                'Use: setup-review --repo-root "$REPO_ROOT" --summary "..."'
            )
        if not re.search(r"--summary", command):
            output_block(
                "BLOCKED: setup-review requires --summary flag. "
                'Use: setup-review --repo-root "$REPO_ROOT" --summary "..."'
            )

    # Validate select-add has --window and --tab
    if "select-add" in command:
        if not re.search(r"--window", command):
            output_block(
                "BLOCKED: select-add requires --window flag. "
                'Use: select-add --window "$W" --tab "$T" <path>'
            )

    # Enforce flowctl done requires --evidence-json and --summary-file
    if " done " in command and ("flowctl" in command or "FLOWCTL" in command):
        # Skip if it's just "flowctl done --help" or similar
        if not re.search(r"--help|-h", command):
            if not re.search(r"--evidence-json|--evidence", command):
                output_block(
                    "BLOCKED: flowctl done requires --evidence-json flag. "
                    "You must capture commit SHAs and test commands. "
                    "Use: flowctl done <task> --summary-file <s.md> --evidence-json <e.json>"
                )
            if not re.search(r"--summary-file|--summary", command):
                output_block(
                    "BLOCKED: flowctl done requires --summary-file flag. "
                    "You must write a done summary. "
                    "Use: flowctl done <task> --summary-file <s.md> --evidence-json <e.json>"
                )

    # Block receipt writes unless chat-send has succeeded + validate format
    receipt_path = os.environ.get("REVIEW_RECEIPT_PATH", "")
    if receipt_path:
        # Check if this command writes to a receipt path
        receipt_dir = os.path.dirname(receipt_path)
        is_receipt_write = receipt_dir and (
            re.search(rf">\s*['\"]?{re.escape(receipt_dir)}", command)
            or re.search(r">\s*['\"]?.*receipts/.*\.json", command)
            or re.search(r"cat\s*>\s*.*receipt", command, re.I)
        )
        if is_receipt_write:
            state = load_state(session_id)
            if not state.get("chat_send_succeeded") and not state.get(
                "codex_review_succeeded"
            ):
                output_block(
                    "BLOCKED: Cannot write receipt before review completes. "
                    "You must run 'flowctl rp chat-send' or 'flowctl codex impl-review/plan-review' "
                    "and receive a review response before writing the receipt."
                )
            # Validate receipt has required 'id' field
            if '"id"' not in command and "'id'" not in command:
                output_block(
                    "BLOCKED: Receipt JSON is missing required 'id' field. "
                    'Receipt must include: {"type":"...","id":"<TASK_OR_EPIC_ID>",...} '
                    "Copy the exact command from the prompt template."
                )
            # Validate completion_review receipts have verdict field
            if "completion_review" in command or "completion-" in receipt_path:
                if '"verdict"' not in command and "'verdict'" not in command:
                    output_block(
                        "BLOCKED: Receipt JSON is missing required 'verdict' field. "
                        'Completion review receipts must include: {"verdict":"SHIP",...} '
                        "Copy the exact command from the prompt template."
                    )
            # For impl receipts, verify flowctl done was called
            if "impl_review" in command:
                # Extract task id from receipt
                id_match = re.search(r'"id"\s*:\s*"([^"]+)"', command)
                if id_match:
                    task_id = id_match.group(1)
                    done_set = state.get("flowctl_done_called", set())
                    if isinstance(done_set, list):
                        done_set = set(done_set)
                    if task_id not in done_set:
                        output_block(
                            f"BLOCKED: Cannot write impl receipt for {task_id} - flowctl done was not called. "
                            f"You MUST run 'flowctl done {task_id} --evidence ...' BEFORE writing the receipt. "
                            "The task is NOT complete until flowctl done succeeds."
                        )

    # All checks passed
    sys.exit(0)


def parse_receipt_path(receipt_path: str) -> tuple:
    """Parse receipt path to derive type and id.

    Returns (receipt_type, item_id) based on filename pattern:
    - plan-fn-N.json or plan-fn-N-xxx.json or plan-fn-N-slug.json
      -> ("plan_review", "fn-N" or "fn-N-xxx" or "fn-N-slug")
    - impl-fn-N.M.json or impl-fn-N-xxx.M.json or impl-fn-N-slug.M.json
      -> ("impl_review", "fn-N.M" or "fn-N-xxx.M" or "fn-N-slug.M")
    - completion-fn-N.json or completion-fn-N-xxx.json or completion-fn-N-slug.json
      -> ("completion_review", "fn-N" or "fn-N-xxx" or "fn-N-slug")

    Suffix pattern supports:
    - Legacy: fn-N (no suffix)
    - Short: fn-N-xxx (1-3 char random)
    - Slug: fn-N-longer-slug (multi-segment slugified title)
    """
    basename = os.path.basename(receipt_path)
    # Suffix pattern: optional hyphen + alphanumeric slug (1-3 char or multi-segment)
    # Pattern: (?:-[a-z0-9][a-z0-9-]*[a-z0-9]|-[a-z0-9]{1,3})?
    suffix_pattern = r"(?:-[a-z0-9][a-z0-9-]*[a-z0-9]|-[a-z0-9]{1,3})?"

    # Try plan pattern first: plan-fn-N.json, plan-fn-N-xxx.json, plan-fn-N-slug.json
    plan_match = re.match(rf"plan-(fn-\d+{suffix_pattern})\.json$", basename)
    if plan_match:
        return ("plan_review", plan_match.group(1))
    # Try impl pattern: impl-fn-N.M.json, impl-fn-N-xxx.M.json, impl-fn-N-slug.M.json
    impl_match = re.match(rf"impl-(fn-\d+{suffix_pattern}\.\d+)\.json$", basename)
    if impl_match:
        return ("impl_review", impl_match.group(1))
    # Try completion pattern: completion-fn-N.json, completion-fn-N-xxx.json, etc.
    completion_match = re.match(rf"completion-(fn-\d+{suffix_pattern})\.json$", basename)
    if completion_match:
        return ("completion_review", completion_match.group(1))
    # Fallback
    return ("impl_review", "UNKNOWN")


def handle_post_tool_use(data: dict) -> None:
    """Handle PostToolUse event - track state and provide feedback."""
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", {})
    command = tool_input.get("command", "")
    session_id = data.get("session_id", "unknown")

    # Get response text
    response_text = ""
    if isinstance(tool_response, dict):
        response_text = tool_response.get("stdout", "") or str(tool_response)
    elif isinstance(tool_response, str):
        response_text = tool_response

    state = load_state(session_id)

    # Track chat-send calls - must have actual review text, not null
    if "chat-send" in command:
        # Check for successful chat (has "Chat Send" and review text, not null)
        if "Chat Send" in response_text and '{"chat": null}' not in response_text:
            state["chats_sent"] = state.get("chats_sent", 0) + 1
            state["chat_send_succeeded"] = True
            save_state(session_id, state)
        elif '{"chat": null}' in response_text or '{"chat":null}' in response_text:
            # Failed - --json was used incorrectly
            state["chat_send_succeeded"] = False
            save_state(session_id, state)

    # Track codex review calls - check for verdict in output
    if (
        "flowctl" in command
        and "codex" in command
        and ("impl-review" in command or "plan-review" in command or "completion-review" in command)
    ):
        # Codex writes receipt automatically with --receipt flag, but we still track success
        verdict_in_output = re.search(
            r"<verdict>(SHIP|NEEDS_WORK|MAJOR_RETHINK)</verdict>", response_text
        )
        if verdict_in_output:
            state["codex_review_succeeded"] = True
            state["last_verdict"] = verdict_in_output.group(1)
            save_state(session_id, state)

    # Track flowctl done calls - match various invocation patterns:
    # - flowctl done <task>
    # - flowctl.py done <task>
    # - .flow/bin/flowctl done <task>
    # - scripts/ralph/flowctl done <task>
    # - $FLOWCTL done <task>
    # - "$FLOWCTL" done <task>
    if " done " in command and ("flowctl" in command or "FLOWCTL" in command):
        # Debug logging
        with Path("/tmp/ralph-guard-debug.log").open("a") as f:
            f.write(f"  -> flowctl done detected in: {command[:100]}...\n")

        # Extract task ID from command - look for "done" followed by task ID
        # Simplified: just find "done <task_id>" pattern since we already validated flowctl context
        done_match = re.search(r"\bdone\s+([a-zA-Z0-9][a-zA-Z0-9._-]*)", command)
        if done_match:
            task_id = done_match.group(1)
            with Path("/tmp/ralph-guard-debug.log").open("a") as f:
                f.write(
                    f"  -> Extracted task_id: {task_id}, response has 'status': {'status' in response_text.lower()}\n"
                )

            # Check response indicates success (has "status", "done", "updated", or "completed")
            response_lower = response_text.lower()
            if (
                "status" in response_lower
                or "done" in response_lower
                or "updated" in response_lower
                or "completed" in response_lower
            ):
                done_set = state.get("flowctl_done_called", set())
                if isinstance(done_set, list):
                    done_set = set(done_set)
                done_set.add(task_id)
                state["flowctl_done_called"] = done_set
                save_state(session_id, state)
                with Path("/tmp/ralph-guard-debug.log").open("a") as f:
                    f.write(
                        f"  -> Added {task_id} to flowctl_done_called: {done_set}\n"
                    )

    # Track receipt writes - reset review state after write
    receipt_path = os.environ.get("REVIEW_RECEIPT_PATH", "")
    if receipt_path and receipt_path in command and ">" in command:
        state["chat_send_succeeded"] = False  # Reset for next review
        state["codex_review_succeeded"] = False  # Reset codex state too
        save_state(session_id, state)

    # Track setup-review output (W= T=)
    if "setup-review" in command:
        w_match = re.search(r"W=(\d+)", response_text)
        t_match = re.search(r"T=([A-F0-9-]+)", response_text, re.I)
        if w_match:
            state["window"] = w_match.group(1)
        if t_match:
            state["tab"] = t_match.group(1)
        save_state(session_id, state)

    # Check for verdict in response
    verdict_match = re.search(
        r"<verdict>(SHIP|NEEDS_WORK|MAJOR_RETHINK)</verdict>", response_text
    )
    if verdict_match:
        state["last_verdict"] = verdict_match.group(1)
        save_state(session_id, state)

        # If SHIP, remind about receipt (only for rp mode - codex writes receipt automatically)
        if verdict_match.group(1) == "SHIP":
            receipt_path = os.environ.get("REVIEW_RECEIPT_PATH", "")
            # Only remind if receipt doesn't exist and we're in rp mode (not codex)
            if (
                receipt_path
                and not Path(receipt_path).exists()
                and state.get("chat_send_succeeded")
            ):
                # Derive type and id from receipt path
                receipt_type, item_id = parse_receipt_path(receipt_path)
                # Build command with ts variable to avoid shell substitution in JSON
                cmd = (
                    f"mkdir -p \"$(dirname '{receipt_path}')\"\n"
                    'ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
                    f"cat > '{receipt_path}' <<EOF\n"
                    f'{{"type":"{receipt_type}","id":"{item_id}","mode":"rp","verdict":"SHIP","timestamp":"$ts"}}\n'
                    "EOF"
                )
                # Provide feedback to Claude (rp mode only - codex writes receipt automatically)
                output_json(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": (
                                f"IMPORTANT: SHIP verdict received. You MUST now write the receipt. "
                                f"Run this command:\n{cmd}"
                            ),
                        }
                    }
                )

        # Prompt Claude to capture learnings from NEEDS_WORK/MAJOR_RETHINK
        elif verdict_match.group(1) in ("NEEDS_WORK", "MAJOR_RETHINK"):
            if is_memory_enabled():
                output_json(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": (
                                "MEMORY: Review returned NEEDS_WORK. After fixing, consider if any lessons are "
                                "GENERALIZABLE (apply beyond this task). If so, capture with:\n"
                                '  flowctl memory add --type <type> "<one-line lesson>"\n'
                                "Types: pitfall (gotchas/mistakes), convention (patterns to follow), decision (architectural choices)\n"
                                "Skip: task-specific fixes, typos, style issues, or 'fine as-is' explanations."
                            ),
                        }
                    }
                )

    elif "chat-send" in command and "Chat Send" in response_text:
        # chat-send returned but no verdict tag found
        # Check for informal approvals that should have been verdict tags
        if re.search(
            r"\bLGTM\b|\bLooks good\b|\bApproved\b|\bNo issues\b", response_text, re.I
        ):
            output_json(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": (
                            "WARNING: Reviewer responded with informal approval (LGTM/Looks good) "
                            "but did NOT use the required <verdict>SHIP</verdict> tag. "
                            "This means your review prompt was incorrect. "
                            "You MUST use /flow-next:impl-review skill which has the correct prompt format. "
                            "Do NOT improvise review prompts. Re-invoke the skill and try again."
                        ),
                    }
                }
            )

    # Check for {"chat": null} which indicates --json was used incorrectly
    if '{"chat":' in response_text or '{"chat": ' in response_text:
        if "null" in response_text:
            output_json(
                {
                    "decision": "block",
                    "reason": (
                        'ERROR: chat-send returned {"chat": null} which means --json was used. '
                        "This suppresses the review text. Re-run without --json flag."
                    ),
                }
            )

    sys.exit(0)


def handle_stop(data: dict) -> None:
    """Handle Stop event - verify receipt written before allowing stop."""
    session_id = data.get("session_id", "unknown")
    stop_hook_active = data.get("stop_hook_active", False)

    # Prevent infinite loops
    if stop_hook_active:
        sys.exit(0)

    receipt_path = os.environ.get("REVIEW_RECEIPT_PATH", "")

    if receipt_path:
        if not Path(receipt_path).exists():
            # Derive type and id from receipt path
            receipt_type, item_id = parse_receipt_path(receipt_path)
            # Tell worker to invoke the review skill, not write receipt manually
            if receipt_type == "impl_review":
                skill = "/flow-next:impl-review"
                skill_desc = "implementation review"
            elif receipt_type == "completion_review":
                skill = "/flow-next:epic-review"
                skill_desc = "epic completion review"
            else:
                skill = "/flow-next:plan-review"
                skill_desc = "plan review"
            # Block stop - review not completed
            output_json(
                {
                    "decision": "block",
                    "reason": (
                        f"Cannot stop: {skill_desc} not completed.\n"
                        f"You MUST invoke `{skill} {item_id}` to complete the review.\n"
                        f"The skill writes the receipt on SHIP verdict.\n"
                        f"Do NOT write the receipt manually - that skips the actual review."
                    ),
                }
            )

    # Clean up state file
    state_file = get_state_file(session_id)
    if state_file.exists():
        state_file.unlink()

    sys.exit(0)


def handle_subagent_stop(data: dict) -> None:
    """Handle SubagentStop event - same as Stop for subagents."""
    handle_stop(data)


def main():
    # Debug logging - always write to see if hook is being called
    debug_file = Path("/tmp/ralph-guard-debug.log")
    with debug_file.open("a") as f:
        f.write(f"[{os.environ.get('FLOW_RALPH', 'unset')}] Hook called\n")

    # Early exit if not in Ralph mode - no output, no context pollution
    if os.environ.get("FLOW_RALPH") != "1":
        with debug_file.open("a") as f:
            f.write("  -> Exiting: FLOW_RALPH not set to 1\n")
        sys.exit(0)

    # Read input
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        with debug_file.open("a") as f:
            f.write("  -> Exiting: JSON decode error\n")
        sys.exit(0)

    event = data.get("hook_event_name", "")
    tool_name = data.get("tool_name", "")

    with debug_file.open("a") as f:
        f.write(f"  -> Event: {event}, Tool: {tool_name}\n")

    # Only process Bash tool calls for Pre/Post
    if event in ("PreToolUse", "PostToolUse") and tool_name != "Bash":
        with debug_file.open("a") as f:
            f.write("  -> Skipping: not Bash\n")
        sys.exit(0)

    # Route to handler
    if event == "PreToolUse":
        handle_pre_tool_use(data)
    elif event == "PostToolUse":
        handle_post_tool_use(data)
    elif event == "Stop":
        handle_stop(data)
    elif event == "SubagentStop":
        handle_subagent_stop(data)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
