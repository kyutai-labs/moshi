#!/usr/bin/env python3
"""
Watch filter for Ralph - parses Claude's stream-json output and shows key events.

Reads JSON lines from stdin, outputs formatted tool calls in TUI style.

CRITICAL: This filter is "fail open" - if output breaks, it continues draining
stdin to prevent SIGPIPE cascading to upstream processes (tee, claude).

Usage:
    watch-filter.py           # Show tool calls only
    watch-filter.py --verbose # Show tool calls + thinking + text responses
"""

import argparse
import json
import os
import sys
from typing import Optional

# Global flag to disable output on pipe errors (fail open pattern)
_output_disabled = False

# ANSI color codes (match ralph.sh TUI)
if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
    C_RESET = "\033[0m"
    C_DIM = "\033[2m"
    C_CYAN = "\033[36m"
else:
    C_RESET = C_DIM = C_CYAN = ""

# TUI indentation (3 spaces to match ralph.sh)
INDENT = "   "

# Tool icons
ICONS = {
    "Bash": "🔧",
    "Edit": "📝",
    "Write": "📄",
    "Read": "📖",
    "Grep": "🔍",
    "Glob": "📁",
    "Task": "🤖",
    "WebFetch": "🌐",
    "WebSearch": "🔎",
    "TodoWrite": "📋",
    "AskUserQuestion": "❓",
    "Skill": "⚡",
}


def safe_print(msg: str) -> None:
    """Print that fails open - disables output on BrokenPipe instead of crashing."""
    global _output_disabled
    if _output_disabled:
        return
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        _output_disabled = True


def drain_stdin() -> None:
    """Consume remaining stdin to prevent SIGPIPE to upstream processes."""
    try:
        for _ in sys.stdin:
            pass
    except Exception:
        pass


def truncate(s: str, max_len: int = 60) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def format_tool_use(tool_name: str, tool_input: dict) -> str:
    """Format a tool use event for TUI display."""
    icon = ICONS.get(tool_name, "🔹")

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        if desc:
            return f"{icon} Bash: {truncate(desc)}"
        return f"{icon} Bash: {truncate(cmd, 60)}"

    elif tool_name == "Edit":
        path = tool_input.get("file_path", "")
        return f"{icon} Edit: {path.split('/')[-1] if path else 'unknown'}"

    elif tool_name == "Write":
        path = tool_input.get("file_path", "")
        return f"{icon} Write: {path.split('/')[-1] if path else 'unknown'}"

    elif tool_name == "Read":
        path = tool_input.get("file_path", "")
        return f"{icon} Read: {path.split('/')[-1] if path else 'unknown'}"

    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        return f"{icon} Grep: {truncate(pattern, 40)}"

    elif tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        return f"{icon} Glob: {pattern}"

    elif tool_name == "Task":
        desc = tool_input.get("description", "")
        agent = tool_input.get("subagent_type", "")
        return f"{icon} Task ({agent}): {truncate(desc, 50)}"

    elif tool_name == "Skill":
        skill = tool_input.get("skill", "")
        return f"{icon} Skill: {skill}"

    elif tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        in_progress = [t for t in todos if t.get("status") == "in_progress"]
        if in_progress:
            return f"{icon} Todo: {truncate(in_progress[0].get('content', ''))}"
        return f"{icon} Todo: {len(todos)} items"

    else:
        return f"{icon} {tool_name}"


def format_tool_result(block: dict) -> Optional[str]:
    """Format a tool_result block (errors only).

    Args:
        block: The full tool_result block (not just content)
    """
    # Check is_error on the block itself
    if block.get("is_error"):
        content = block.get("content", "")
        error_text = str(content) if content else "unknown error"
        return f"{INDENT}{C_DIM}❌ {truncate(error_text, 60)}{C_RESET}"

    # Also check content for error strings (heuristic)
    content = block.get("content", "")
    if isinstance(content, str):
        lower = content.lower()
        if "error" in lower or "failed" in lower:
            return f"{INDENT}{C_DIM}⚠️  {truncate(content, 60)}{C_RESET}"

    return None


def process_event(event: dict, verbose: bool) -> None:
    """Process a single stream-json event."""
    event_type = event.get("type", "")

    # Tool use events (assistant messages)
    if event_type == "assistant":
        message = event.get("message", {})
        content = message.get("content", [])

        for block in content:
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                formatted = format_tool_use(tool_name, tool_input)
                safe_print(f"{INDENT}{C_DIM}{formatted}{C_RESET}")

            elif verbose and block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    safe_print(f"{INDENT}{C_CYAN}💬 {text}{C_RESET}")

            elif verbose and block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking.strip():
                    safe_print(f"{INDENT}{C_DIM}🧠 {truncate(thinking, 100)}{C_RESET}")

    # Tool results (user messages with tool_result blocks)
    elif event_type == "user":
        message = event.get("message", {})
        content = message.get("content", [])

        for block in content:
            if block.get("type") == "tool_result":
                formatted = format_tool_result(block)
                if formatted:
                    safe_print(formatted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Claude stream-json output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show text and thinking in addition to tool calls",
    )
    args = parser.parse_args()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        try:
            process_event(event, args.verbose)
        except Exception:
            # Swallow processing errors - keep draining stdin
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except BrokenPipeError:
        # Output broken but keep draining to prevent upstream SIGPIPE
        drain_stdin()
        sys.exit(0)
    except Exception as e:
        print(f"watch-filter: {e}", file=sys.stderr)
        drain_stdin()
        sys.exit(0)
