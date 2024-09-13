# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import sys


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    if level == "warning":
        prefix = colorize("Warning:", "1;31")
    elif level == "info":
        prefix = colorize("Info:", "1;34")
    elif level == "error":
        prefix = colorize("Error:", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + " " + msg


class RawPrinter:
    def __init__(self, stream=sys.stdout, err_stream=sys.stderr):
        self.stream = stream
        self.err_stream = err_stream

    def print_header(self):
        pass

    def print_token(self, token: str):
        self.stream.write(token)
        self.stream.flush()

    def log(self, level: str, msg: str):
        print(f"{level.capitalize()}: {msg}", file=self.err_stream)

    def print_lag(self):
        self.err_stream.write(colorize(" [LAG]", "31"))
        self.err_stream.flush()

    def print_pending(self):
        pass


@dataclass
class LineEntry:
    msg: str
    color: str | None = None

    def render(self):
        if self.color is None:
            return self.msg
        else:
            return colorize(self.msg, self.color)

    def __len__(self):
        return len(self.msg)


class Line:
    def __init__(self, stream):
        self.stream = stream
        self._line: list[LineEntry] = []
        self._has_padding: bool = False
        self._max_line_length = 0

    def __bool__(self):
        return bool(self._line)

    def __len__(self):
        return sum(len(entry) for entry in self._line)

    def add(self, msg: str, color: str | None = None) -> int:
        entry = LineEntry(msg, color)
        return self._add(entry)

    def _add(self, entry: LineEntry) -> int:
        if self._has_padding:
            self.erase(count=0)
        self._line.append(entry)
        self.stream.write(entry.render())
        self._max_line_length = max(self._max_line_length, len(self))
        return len(entry)

    def erase(self, count: int = 1):
        if count:
            entries = list(self._line[:-count])
        else:
            entries = list(self._line)
        self._line.clear()
        self.stream.write("\r")
        for entry in entries:
            self._line.append(entry)
            self.stream.write(entry.render())

        self._has_padding = False

    def newline(self):
        missing = self._max_line_length - len(self)
        if missing > 0:
            self.stream.write(" " * missing)
        self.stream.write("\n")
        self._line.clear()
        self._max_line_length = 0
        self._has_padding = False

    def flush(self):
        missing = self._max_line_length - len(self)
        if missing > 0:
            self.stream.write(" " * missing)
            self._has_padding = True
        self.stream.flush()


class Printer:
    def __init__(self, max_cols: int = 80, stream=sys.stdout, err_stream=sys.stderr):
        self.max_cols = max_cols
        self.line = Line(stream)
        self.stream = stream
        self.err_stream = err_stream
        self._pending_count = 0
        self._pending_printed = False

    def print_header(self):
        self.line.add(" " + "-" * (self.max_cols) + " ")
        self.line.newline()
        self.line.flush()
        self.line.add("| ")

    def _remove_pending(self) -> bool:
        if self._pending_printed:
            self._pending_printed = False
            self.line.erase(1)
            return True
        return False

    def print_token(self, token: str, color: str | None = None):
        self._remove_pending()
        remaining = self.max_cols - len(self.line)
        if len(token) <= remaining:
            self.line.add(token, color)
        else:
            end = " " * remaining + " |"
            if token.startswith(" "):
                token = token.lstrip()
                self.line.add(end)
                self.line.newline()
                self.line.add("| ")
                self.line.add(token, color)
            else:
                assert color is None
                erase_count = None
                cumulated = ""
                for idx, entry in enumerate(self.line._line[::-1]):
                    if entry.color:
                        # probably a LAG message
                        erase_count = idx
                        break
                    if entry.msg.startswith(" "):
                        erase_count = idx + 1
                        cumulated = entry.msg + cumulated
                        break
                if erase_count is not None:
                    if erase_count > 0:
                        self.line.erase(erase_count)
                    remaining = self.max_cols - len(self.line)
                    end = " " * remaining + " |"
                    self.line.add(end)
                    self.line.newline()
                    self.line.add("| ")
                    token = cumulated.lstrip() + token
                    self.line.add(token)
                else:
                    self.line.add(token[:remaining])
                    self.line.add(" |")
                    self.line.newline()
                    self.line.add("| ")
                    self.line.add(token[remaining:])
        self.line.flush()

    def log(self, level: str, msg: str):
        msg = make_log(level, msg)
        self._remove_pending()
        if self.line:
            self.line.newline()
        self.line.flush()
        print(msg, file=self.err_stream)
        self.err_stream.flush()

    def print_lag(self):
        self.print_token(" [LAG]", "31")

    def print_pending(self):
        chars = ["|", "/", "-", "\\"]
        count = int(self._pending_count / 5)
        char = chars[count % len(chars)]
        colors = ["32", "33", "31"]
        self._remove_pending()
        self.line.add(char, colors[count % len(colors)])
        self._pending_printed = True
        self._pending_count += 1


AnyPrinter = Printer | RawPrinter
