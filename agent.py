#!/usr/bin/env python3
"""CLI agent. Two backends, same ReAct loop:
  - LIBRECHAT_URL set  -> talks to LibreChat's REST API using cookies from session.json
  - OPENAI_BASE_URL set -> talks to any OpenAI-compatible endpoint (e.g. local Ollama)
ReAct prompting (Qwen2.5 native <tool_call> format) — works regardless of whether
the backend supports OpenAI tool definitions.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

# tolerate bad bytes from terminal pastes / non-UTF-8 locales
for s in (sys.stdin, sys.stdout):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _try_json(r):
    """Decode response as JSON, return None on failure (never raises)."""
    try:
        return r.json()
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_url(url, name="URL"):
    """Return (parsed, error_msg). error_msg is empty string on success."""
    if not url:
        return None, f"{name} is empty"
    from urllib.parse import urlparse
    try:
        p = urlparse(url)
    except (ValueError, AttributeError) as e:
        return None, f"{name} is malformed: {e}"
    if p.scheme not in ("http", "https"):
        return None, f"{name} must start with http:// or https:// (got {p.scheme or 'no scheme'})"
    if not p.hostname:
        return None, f"{name} has no hostname"
    return p, ""


def _validate_session(state):
    """Return error_msg ('' if session is usable)."""
    if not isinstance(state, dict):
        return "session file is not a JSON object"
    if not isinstance(state.get("token"), str) or not state["token"]:
        return "session has no JWT — run a login command"
    cookies = state.get("cookies")
    if not isinstance(cookies, list):
        return "session has no cookies array"
    rt = next((c for c in cookies if isinstance(c, dict) and c.get("name") == "refreshToken"), None)
    if not rt or not isinstance(rt.get("value"), str) or not rt["value"]:
        return "session has no refreshToken cookie — run a login command"
    return ""


# transient HTTP failures we retry (with bounded backoff)
_RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                     httpx.PoolTimeout, httpx.RemoteProtocolError)


def _request_with_retry(send_fn, *, attempts=3, base_delay=0.5):
    """Run send_fn() (returning httpx.Response) with bounded backoff on
    transient errors. Caller owns interpretation of non-retried responses."""
    last_exc = None
    for i in range(attempts):
        try:
            r = send_fn()
        except _RETRY_EXCEPTIONS as e:
            last_exc = e
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
            continue
        if r.status_code in _RETRY_STATUSES and i < attempts - 1:
            time.sleep(base_delay * (2 ** i))
            continue
        return r
    raise last_exc if last_exc else RuntimeError("retry exhausted without response")

CONFIG_DIR = Path(os.environ.get("CLI_AGENT_CONFIG_DIR") or
                  Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "cli-agent")
_env_file = CONFIG_DIR / ".env"
load_dotenv(_env_file if _env_file.exists() else None)

LIBRECHAT_URL = os.environ.get("LIBRECHAT_URL", "").rstrip("/")
LIBRECHAT_MODEL = os.environ.get("LIBRECHAT_MODEL", "")
LIBRECHAT_ENDPOINT = os.environ.get("LIBRECHAT_ENDPOINT", "")  # body field, e.g. "Ollama" or "openAI"
LIBRECHAT_ENDPOINT_TYPE = os.environ.get("LIBRECHAT_ENDPOINT_TYPE", "")  # URL path, e.g. "custom" or "openAI"
SESSION_FILE = os.environ.get("SESSION_FILE", "session.json")
if not Path(SESSION_FILE).is_absolute():
    SESSION_FILE = str(CONFIG_DIR / SESSION_FILE)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:3b")

TOOL_REMINDER = (
    "REMINDER: you have real tools that execute on the user's machine. "
    "To do anything, emit a <tool_call>...</tool_call> block. "
    "DO NOT write bash code blocks for the user to copy-run — they will not be executed. "
    "DO NOT just describe steps. Call the tool yourself."
)

SYSTEM_PROMPT = (
    "You are a CLI coding agent in the user's terminal.\n"
    f"Your current working directory is: {os.getcwd()}\n"
    'When the user says "this folder", "here", or uses relative paths, they mean\n'
    'this directory. Do not list "/" unless explicitly asked.\n\n'
) + """Available tools:
  - read_file(path: str)
  - write_file(path: str, content: str)              # creates or overwrites
  - edit_file(path: str, old_string: str, new_string: str, replace_all?: bool)
       finds old_string in the file (must be unique unless replace_all=true)
       and replaces it with new_string. Use this for small/targeted edits
       instead of rewriting the whole file with write_file.
  - list_dir(path: str)
  - run_bash(command: str, background?: bool)
       background=true returns immediately with a job_id — use it for servers,
       watchers, builds, or anything that won't finish in seconds. Default
       (background=false) blocks for up to 60s.
  - monitor_bash(job_id: str, tail_lines?: int)      # check one bg job
  - list_bg_jobs()                                   # list all bg jobs in this session
  - kill_bash(job_id: str)                           # stop a bg job

You can interrupt a foreground run_bash by pressing Ctrl-C in the terminal —
it will be killed and you'll be told what partial output was captured.

To call a tool, emit a block in EXACTLY this shape (nothing before or after):

<tool_call>
{"name": "read_file", "arguments": {"path": "/etc/hostname"}}
</tool_call>

Replace "read_file" with the actual tool you need and the arguments with real
values. The whole inside MUST be a single valid JSON object with exactly two
keys: "name" (string) and "arguments" (object). Then STOP — the user will
reply with the result wrapped in <tool_response>...</tool_response>.

DO NOT use any other format. DO NOT write `TOOL_NAME: …` lines or YAML or
plain text labels — only the <tool_call> block above will be executed.
DO NOT write bash code blocks for the user to run; emit a tool_call with
run_bash instead. After the result comes back, call another tool or write
the final plain-text answer (no tool_call block in that case).

Be concise. Stop and ask before destructive actions (rm -rf, dropping data, force pushes)."""


# --- tools ---
def tool_read_file(path):
    return Path(path).expanduser().read_text()

def tool_write_file(path, content):
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"

def tool_edit_file(path, old_string, new_string, replace_all=False):
    p = Path(path).expanduser()
    text = p.read_text()
    count = text.count(old_string)
    if count == 0:
        return f"ERROR: old_string not found in {p}"
    if count > 1 and not replace_all:
        return (f"ERROR: old_string appears {count} times in {p} — provide more "
                "surrounding context to make it unique, or pass replace_all=true")
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    p.write_text(new_text)
    n = count if replace_all else 1
    return f"edited {p} ({n} replacement{'s' if n != 1 else ''})"

def tool_list_dir(path):
    entries = sorted(os.listdir(Path(path).expanduser()))
    return "\n".join(entries) if entries else "(empty)"

_BG_JOBS_DIR = Path("/tmp/cli-agent-jobs")
_BG_JOBS = {}  # job_id -> {"proc": Popen, "log": Path, "command": str, "started": float}
_LAST_FG_LOG = [None]  # Path | None; set after each foreground run_bash

# tracks the *currently-running* foreground command so keybindings can act on it
_CURRENT_FG = {"proc": None, "log": None, "command": None, "started": 0.0}
_FG_BG_REQUESTED = threading.Event()  # set by Ctrl+B; _run_fg detaches when it sees it


def _tail_file(path, n):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"(error reading log: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"


def _cap_for_model(text, max_lines=2000, max_bytes=200_000, log_path=None):
    """Trim live output before it goes back to the model. Always include the
    last lines (most relevant) and a hint about the log file for full output."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines and len(text) <= max_bytes:
        return text
    head = "".join(lines[:max_lines // 2])
    tail = "".join(lines[-max_lines // 2:])
    note = (f"\n--- output truncated ({len(lines)} lines, {len(text)} bytes) "
            f"— full log at {log_path} ---\n" if log_path else
            f"\n--- output truncated ({len(lines)} lines) ---\n")
    return head + note + tail


def _run_fg(command):
    """Foreground bash: stream stdout+stderr to the terminal live, save full
    output to a log file, return a (capped) result string for the model.
    If Ctrl+B is pressed (sets _FG_BG_REQUESTED), detach and continue in bg."""
    _BG_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _BG_JOBS_DIR / f"fg-{int(time.time())}-{uuid.uuid4().hex[:6]}.log"
    _LAST_FG_LOG[0] = log_path

    proc = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured = []
    log_fd = log_path.open("w")
    timed_out = [False]
    backgrounded = [False]

    _CURRENT_FG.update(proc=proc, log=log_path, command=command, started=time.time())
    _FG_BG_REQUESTED.clear()

    stop_evt = threading.Event()

    def killer():
        if not stop_evt.wait(60):
            if proc.poll() is None and not backgrounded[0]:
                timed_out[0] = True
                try: proc.terminate()
                except Exception: pass

    threading.Thread(target=killer, daemon=True).start()

    def takeover_thread(proc, log_fd):
        """After backgrounding: keep reading stdout, dump to log, until exit."""
        try:
            for line in proc.stdout:
                log_fd.write(line); log_fd.flush()
        except Exception:
            pass
        finally:
            try: log_fd.close()
            except Exception: pass
            try: proc.wait()
            except Exception: pass

    prefix = f"{_C['cyan']}  | {_C['reset']}"
    try:
        for line in proc.stdout:
            if _FG_BG_REQUESTED.is_set():
                # detach: register as bg job, hand the proc + log_fd to takeover thread
                _FG_BG_REQUESTED.clear()
                bg_id = uuid.uuid4().hex[:8]
                _BG_JOBS[bg_id] = {
                    "proc": proc, "log": log_path, "command": command,
                    "started": _CURRENT_FG["started"], "log_fd": log_fd,
                }
                threading.Thread(target=takeover_thread, args=(proc, log_fd), daemon=True).start()
                backgrounded[0] = True
                stop_evt.set()  # tell killer to stop watching this proc
                # write the line that triggered the check (already read from pipe)
                log_fd.write(line); log_fd.flush()
                captured.append(line)
                sys.stdout.write(prefix + line); sys.stdout.flush()
                output = "".join(captured)
                return (f"foreground command MOVED TO BACKGROUND as {bg_id}\n"
                        f"log: {log_path}\n"
                        f"command: {command}\n"
                        f"--- output so far ---\n{_cap_for_model(output, log_path=log_path)}\n"
                        f"--- (still running — use monitor_bash('{bg_id}') / kill_bash) ---")
            sys.stdout.write(prefix + line); sys.stdout.flush()
            captured.append(line)
            log_fd.write(line); log_fd.flush()
    except KeyboardInterrupt:
        stop_evt.set()
        try: proc.terminate()
        except Exception: pass
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: proc.kill()
            except Exception: pass
        log_fd.close()
        _CURRENT_FG.update(proc=None, log=None, command=None)
        raise
    finally:
        if not backgrounded[0]:
            stop_evt.set()
            proc.wait()
            log_fd.close()
            _CURRENT_FG.update(proc=None, log=None, command=None)

    output = "".join(captured)
    if timed_out[0]:
        return (f"ERROR: foreground command timed out after 60s and was killed. "
                f"Use background=true for long-running jobs.\n"
                f"log: {log_path}\n"
                f"--- captured output ---\n{_cap_for_model(output, log_path=log_path)}")
    return (f"exit={proc.returncode}  log={log_path}\n"
            f"--- output ---\n{_cap_for_model(output, log_path=log_path)}")


def tool_run_bash(command, background=False):
    if not background:
        return _run_fg(command)

    # background mode
    import time as _time
    _BG_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    log_path = _BG_JOBS_DIR / f"{job_id}.log"
    log_fd = log_path.open("wb")
    try:
        proc = subprocess.Popen(
            command, shell=True,
            stdout=log_fd, stderr=subprocess.STDOUT,
            start_new_session=True,  # detach so Ctrl-C in agent doesn't kill it
        )
    except OSError as e:
        log_fd.close()
        return f"ERROR: failed to start: {e}"
    _BG_JOBS[job_id] = {"proc": proc, "log": log_path, "command": command,
                        "started": _time.time(), "log_fd": log_fd}
    return (f"started background job {job_id} (pid {proc.pid})\n"
            f"log: {log_path}\n"
            f"call monitor_bash(job_id='{job_id}') to check status, "
            f"kill_bash(job_id='{job_id}') to stop it.")


def tool_monitor_bash(job_id, tail_lines=50):
    log_path = _BG_JOBS_DIR / f"{job_id}.log"
    job = _BG_JOBS.get(job_id)
    if not job:
        if log_path.exists():
            return (f"job {job_id} not tracked in this session (agent restarted?) — "
                    f"showing log file:\n{_tail_file(log_path, tail_lines)}")
        return f"ERROR: no such job '{job_id}'"
    rc = job["proc"].poll()
    import time as _time
    elapsed = int(_time.time() - job["started"])
    status = "running" if rc is None else f"exited (rc={rc})"
    return (f"job {job_id}: {status} ({elapsed}s elapsed)\n"
            f"command: {job['command']}\n"
            f"--- last {tail_lines} log lines ---\n"
            f"{_tail_file(log_path, tail_lines)}")


def tool_list_bg_jobs():
    if not _BG_JOBS:
        return "no background jobs"
    import time as _time
    lines = []
    for jid, job in _BG_JOBS.items():
        rc = job["proc"].poll()
        status = "running" if rc is None else f"exited (rc={rc})"
        elapsed = int(_time.time() - job["started"])
        cmd = job["command"]
        lines.append(f"  {jid}  {status:<18}  {elapsed:>4}s  {cmd[:80]}")
    return "background jobs:\n" + "\n".join(lines)


def tool_kill_bash(job_id):
    job = _BG_JOBS.get(job_id)
    if not job:
        return f"ERROR: no such job '{job_id}'"
    proc = job["proc"]
    if proc.poll() is not None:
        return f"job {job_id} already exited (rc={proc.returncode})"
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    rc = proc.returncode
    return f"killed job {job_id} (pid {proc.pid}, rc={rc})"

DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "run_bash": tool_run_bash,
    "monitor_bash": tool_monitor_bash,
    "list_bg_jobs": tool_list_bg_jobs,
    "kill_bash": tool_kill_bash,
}

# schema per tool: required + optional arg types. Extra args are dropped silently.
_TOOL_SCHEMAS = {
    "read_file":    {"req": {"path": str},                                                      "opt": {}},
    "write_file":   {"req": {"path": str, "content": str},                                      "opt": {}},
    "edit_file":    {"req": {"path": str, "old_string": str, "new_string": str},                "opt": {"replace_all": bool}},
    "list_dir":     {"req": {"path": str},                                                      "opt": {}},
    "run_bash":     {"req": {"command": str},                                                   "opt": {"background": bool}},
    "monitor_bash": {"req": {"job_id": str},                                                    "opt": {"tail_lines": int}},
    "list_bg_jobs": {"req": {},                                                                 "opt": {}},
    "kill_bash":    {"req": {"job_id": str},                                                    "opt": {}},
}


def _validate_tool_args(name, args):
    if name not in DISPATCH:
        return f"unknown tool '{name}' (available: {', '.join(DISPATCH)})"
    if not isinstance(args, dict):
        return f"args must be a JSON object, got {type(args).__name__}"
    schema = _TOOL_SCHEMAS[name]
    for k, t in schema["req"].items():
        if k not in args:
            return f"missing required arg '{k}' for {name}"
        if not isinstance(args[k], t):
            return f"arg '{k}' for {name} must be {t.__name__}, got {type(args[k]).__name__}"
        if t is str and not args[k]:
            return f"arg '{k}' for {name} must be non-empty"
    for k, t in schema["opt"].items():
        if k in args and not isinstance(args[k], t):
            return f"optional arg '{k}' for {name} must be {t.__name__}, got {type(args[k]).__name__}"
    return None


def call_tool(name, args):
    err = _validate_tool_args(name, args)
    if err:
        return f"ERROR: {err}"
    schema = _TOOL_SCHEMAS[name]
    valid_keys = set(schema["req"]) | set(schema["opt"])
    safe_args = {k: v for k, v in args.items() if k in valid_keys}
    try:
        return DISPATCH[name](**safe_args)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# --- ReAct parsing ---
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def _parse_call(raw):
    obj = json.loads(raw)
    args = obj.get("arguments", obj.get("args", {}))
    return obj["name"], args if isinstance(args, dict) else {}

def _find_balanced_json_objects(text):
    """Yield substrings of `text` that are top-level balanced {...} blocks."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start:i+1]
                start = -1

_KV_NAME_RE = re.compile(
    r"(?:^|\n)\s*(?:tool[_-]?name|TOOL_NAME|tool|function|name)\s*[:=]\s*[\"']?([a-zA-Z_]\w*)[\"']?",
    re.IGNORECASE,
)

def _kv_drift_calls(text, valid_names):
    """Last-resort parser for models that drift to 'TOOL_NAME: x / arguments: {...}'
    style instead of emitting <tool_call> JSON. Yields (name, args_dict)."""
    out = []
    for m in _KV_NAME_RE.finditer(text):
        name = m.group(1)
        if name not in valid_names:
            continue
        rest = text[m.end():]
        for raw in _find_balanced_json_objects(rest):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            # could be {arguments: {...}} or the args dict directly
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
            out.append((name, args))
            break
    return out

def extract_tool_calls(text):
    calls = []
    matches = list(TOOL_RE.finditer(text))
    if matches:
        for m in matches:
            try:
                calls.append(_parse_call(m.group(1)))
            except (json.JSONDecodeError, KeyError) as e:
                calls.append(("__PARSE_ERROR__", {"raw": m.group(1), "error": str(e)}))
        return calls
    # fallback 1: bare JSON object with "name" + ("arguments"|"args")
    for raw in _find_balanced_json_objects(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "args" in obj):
            calls.append(_parse_call(raw))
    if calls:
        return calls
    # fallback 2: KV drift ("TOOL_NAME: foo\narguments: {...}")
    return _kv_drift_calls(text, set(DISPATCH))


# --- backends ---
class OpenAIBackend:
    def __init__(self):
        from openai import OpenAI
        # disable cert verification on the underlying httpx client used by the SDK
        self.client = OpenAI(
            base_url=OPENAI_BASE_URL or None,
            api_key=OPENAI_API_KEY,
            http_client=httpx.Client(verify=False, timeout=httpx.Timeout(180.0, connect=10.0)),
        )
        self.model = AGENT_MODEL

    def send_stream(self, messages):
        """Yield text chunks. Caller accumulates."""
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


def _find_token_in_storage(state):
    """Playwright's storage_state() puts localStorage under 'origins'. LibreChat
    stores the JWT under various keys depending on version — scan for it."""
    for origin in state.get("origins", []):
        for item in origin.get("localStorage", []):
            v = item.get("value", "")
            if v.startswith("eyJ"):  # JWTs always start with this
                return v.strip('"')
            try:
                obj = json.loads(v)
                for key in ("token", "accessToken", "jwt"):
                    if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].startswith("eyJ"):
                        return obj[key]
            except (json.JSONDecodeError, TypeError):
                pass
    return None


class LibreChatBackend:
    """Talks to LibreChat's internal REST API using session cookies from login.py.

    LibreChat manages conversation state server-side: we track conversationId and
    parentMessageId between turns. System prompt is prepended to the first user
    message since /api/ask/* doesn't take a separate system message field reliably
    across versions."""

    def __init__(self):
        # validate config before doing anything else
        _, err = _validate_url(LIBRECHAT_URL, "LIBRECHAT_URL")
        if err:
            raise RuntimeError(f"config error: {err}. Run: cli-agent config")
        for name, val in [("LIBRECHAT_MODEL", LIBRECHAT_MODEL), ("LIBRECHAT_ENDPOINT", LIBRECHAT_ENDPOINT)]:
            if not val:
                raise RuntimeError(f"config error: {name} not set. Run: cli-agent config")

        # load + validate session
        if not Path(SESSION_FILE).exists():
            raise RuntimeError(f"no session at {SESSION_FILE} — run a login command first")
        try:
            state = json.loads(Path(SESSION_FILE).read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"session file {SESSION_FILE} is not valid JSON: {e}. "
                               "Re-run a login command.")
        err = _validate_session(state)
        if err:
            raise RuntimeError(f"{err}")
        # plain dict — httpx.Cookies jar's domain matching was silently dropping our
        # cookies. We only need the refreshToken cookie on /api/auth/refresh, so we
        # pass it explicitly there rather than relying on a jar.
        self._cookies = {c["name"]: c["value"]
                         for c in state.get("cookies", [])
                         if isinstance(c, dict) and isinstance(c.get("name"), str)
                         and isinstance(c.get("value"), str)}
        token = state.get("token") or _find_token_in_storage(state)
        if not token:
            raise RuntimeError("no JWT in session.json — re-run login")
        self.client = httpx.Client(
            base_url=LIBRECHAT_URL,
            timeout=httpx.Timeout(180.0, connect=10.0),
            verify=False,  # caller asserts trust — chatouille often uses internal certs
            headers={
                # LibreChat's uaParser middleware rejects non-browser User-Agents
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Authorization": f"Bearer {token}",
            },
        )
        self.conversation_id = None
        self.parent_message_id = "00000000-0000-0000-0000-000000000000"
        self._first_turn = True

    def _refresh_token(self):
        # /api/auth/refresh exchanges the refreshToken cookie for a new access token
        if "refreshToken" not in self._cookies:
            raise RuntimeError("no refreshToken in session — re-run a login command")
        r = _request_with_retry(
            lambda: self.client.post("/api/auth/refresh", cookies=self._cookies))
        if r.status_code != 200:
            raise RuntimeError(
                f"refresh failed: HTTP {r.status_code} — {r.text[:200]}. "
                "The refreshToken is probably expired — re-run a login command.")
        body = _try_json(r)
        new_token = (body or {}).get("token")
        if not new_token:
            raise RuntimeError(
                f"refresh returned unexpected body (HTTP {r.status_code}): "
                f"{r.text[:200]} — re-run a login command.")
        # capture rotated refreshToken if the server issued a new one
        for sc in r.headers.get_list("set-cookie") if hasattr(r.headers, "get_list") else []:
            if sc.startswith("refreshToken="):
                self._cookies["refreshToken"] = sc.split("=", 1)[1].split(";", 1)[0]
                break
        # also check via the multi-value Set-Cookie header (httpx exposes per-line)
        raw = r.headers.get("set-cookie", "")
        if raw and "refreshToken=" in raw:
            for part in raw.split(","):
                part = part.strip()
                if part.startswith("refreshToken="):
                    self._cookies["refreshToken"] = part.split("=", 1)[1].split(";", 1)[0]
                    break
        self.client.headers["Authorization"] = f"Bearer {new_token}"
        # persist
        state = json.loads(Path(SESSION_FILE).read_text())
        state["token"] = new_token
        for c in state.get("cookies", []):
            if c["name"] == "refreshToken":
                c["value"] = self._cookies["refreshToken"]
                break
        Path(SESSION_FILE).write_text(json.dumps(state, indent=2))

    def send_stream(self, messages):
        """Yield text deltas as the model generates. LibreChat events sometimes
        carry the cumulative text rather than a delta — we diff against what we've
        seen so far either way."""
        latest_user = next(m for m in reversed(messages) if m["role"] == "user")
        text = latest_user["content"]
        if self._first_turn and messages and messages[0]["role"] == "system":
            text = messages[0]["content"] + "\n\n---\n\n" + text
            self._first_turn = False

        msg_id = str(uuid.uuid4())
        from datetime import datetime, timezone
        payload = {
            "text": text,
            "sender": "User",
            "clientTimestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "isCreatedByUser": True,
            "parentMessageId": self.parent_message_id,
            "messageId": msg_id,
            "error": False,
            "endpoint": LIBRECHAT_ENDPOINT,
            "model": LIBRECHAT_MODEL,
            "key": "never",
            "isTemporary": False,
            "isRegenerate": False,
            "isContinued": False,
        }
        if LIBRECHAT_ENDPOINT_TYPE:
            payload["endpointType"] = LIBRECHAT_ENDPOINT_TYPE
        if self.conversation_id:
            payload["conversationId"] = self.conversation_id

        url = f"/api/agents/chat/{LIBRECHAT_ENDPOINT}"
        r = _request_with_retry(lambda: self.client.post(url, json=payload))
        if r.status_code == 401:
            self._refresh_token()
            r = _request_with_retry(lambda: self.client.post(url, json=payload))
        if r.status_code != 200:
            raise RuntimeError(
                f"chat POST {url} failed: HTTP {r.status_code} — {r.text[:300]}")
        init = _try_json(r)
        if not init or "streamId" not in init:
            raise RuntimeError(
                f"chat POST {url} returned unexpected body (HTTP {r.status_code}): "
                f"{r.text[:300]}")
        stream_id = init["streamId"]
        self.conversation_id = init.get("conversationId", self.conversation_id)

        seen_text = ""
        final_event = None
        with self.client.stream(
            "GET", f"/api/agents/chat/stream/{stream_id}",
            headers={"Accept": "text/event-stream"},
        ) as sr:
            if sr.status_code != 200:
                raise RuntimeError(
                    f"chat stream failed: HTTP {sr.status_code} — "
                    f"{sr.read().decode('utf-8', 'replace')[:300]}")
            for raw in sr.iter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                try:
                    obj = json.loads(raw[5:].strip())
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("final"):
                    final_event = obj
                    break
                # current LibreChat: {"event": "on_message_delta", "data": {"delta": {"content": [{"type":"text","text":"..."}]}}}
                if obj.get("event") == "on_message_delta":
                    data = obj.get("data") or {}
                    delta = data.get("delta") or {}
                    content = delta.get("content") or []
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                chunk = part.get("text") or ""
                                if chunk:
                                    seen_text += chunk
                                    yield chunk
                    continue
                # legacy: top-level 'text' field (older LibreChat versions / some endpoints)
                t = obj.get("text")
                if isinstance(t, str) and t:
                    if t.startswith(seen_text):
                        d = t[len(seen_text):]
                        seen_text = t
                    else:
                        d = t
                        seen_text += t
                    if d:
                        yield d

        if final_event:
            resp = final_event.get("responseMessage") or {}
            self.parent_message_id = resp.get("messageId", self.parent_message_id)
            final_text = resp.get("text") or ""
            if not final_text:
                for part in resp.get("content", []) or []:
                    if part.get("type") == "text":
                        final_text += part.get("text", "")
                    elif part.get("type") == "error":
                        final_text = f"[LibreChat error] {part.get('error')}"
            # if we streamed less than the final, emit the remainder
            if final_text.startswith(seen_text):
                tail = final_text[len(seen_text):]
                if tail:
                    yield tail
            elif not seen_text and final_text:
                yield final_text


# --- chat session persistence ---
import argparse
import difflib
import signal
import time
import uuid
from datetime import datetime


# --- diff rendering for file edits ---
_ANSI = sys.stdout.isatty()
_C = {"red": "\x1b[31m", "green": "\x1b[32m", "cyan": "\x1b[36m",
      "bold": "\x1b[1m", "reset": "\x1b[0m"} if _ANSI else \
     {k: "" for k in ("red", "green", "cyan", "bold", "reset")}


def _render_diff(old_text, new_text, path, max_lines=120):
    if old_text == new_text:
        return None
    raw = list(difflib.unified_diff(
        old_text.splitlines(keepends=False),
        new_text.splitlines(keepends=False),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3, lineterm="",
    ))
    out = []
    for line in raw:
        if line.startswith("+++") or line.startswith("---"):
            out.append(f"{_C['bold']}{line}{_C['reset']}")
        elif line.startswith("@@"):
            out.append(f"{_C['cyan']}{line}{_C['reset']}")
        elif line.startswith("+"):
            out.append(f"{_C['green']}{line}{_C['reset']}")
        elif line.startswith("-"):
            out.append(f"{_C['red']}{line}{_C['reset']}")
        else:
            out.append(line)
    if len(out) > max_lines:
        out = out[:max_lines] + [f"{_C['cyan']}... ({len(raw) - max_lines} more diff lines){_C['reset']}"]
    return "\n".join(out)


def _show_diff_for(name, ar):
    """Print a diff preview if this tool call would change a file's contents."""
    path_str = ar.get("path") if isinstance(ar, dict) else None
    if not path_str:
        return
    p = Path(path_str).expanduser()
    old = p.read_text() if p.exists() else ""
    if name == "write_file":
        new = ar.get("content", "")
    elif name == "edit_file":
        old_s = ar.get("old_string", "")
        new_s = ar.get("new_string", "")
        if not old_s or old_s not in old:
            return  # tool will error itself; don't pretend to diff
        new = old.replace(old_s, new_s) if ar.get("replace_all") else old.replace(old_s, new_s, 1)
    else:
        return
    diff = _render_diff(old, new, path_str)
    if diff:
        print(diff)

CHATS_DIR = CONFIG_DIR / "chats"

# --- async input queue (filled by prompt_toolkit, drained by turn loop) ---
# Lines submitted while the agent is busy pile up here. The turn loop pulls
# from this queue on each new turn before falling through to a fresh prompt.
_INPUT_QUEUE: "asyncio.Queue[str]" = None  # set in main_async
_AGENT_BUSY = threading.Event()
_RECALL_TEXT = [""]                          # one-shot text to pre-fill prompt

# --- ctrl-c double-tap: single = interrupt, double (within 1.5s) = exit ---
_last_sigint = [0.0]
def _sigint_handler(signum, frame):
    now = time.monotonic()
    if now - _last_sigint[0] < 1.5:
        print("\n(double Ctrl-C — exiting)")
        os._exit(130)
    _last_sigint[0] = now
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, _sigint_handler)

# --- permission modes ---
# safe:      ask before run_bash and write_file
# auto-edit: ask before run_bash, edits go through silently
# yolo:      ask for nothing
_TOOLS_NEEDING_CONFIRM = {
    "safe":      {"run_bash", "write_file", "edit_file", "kill_bash"},
    "auto-edit": {"run_bash", "kill_bash"},
    "yolo":      set(),
}


def _list_chats():
    if not CHATS_DIR.exists():
        return []
    items = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            items.append({
                "id": f.stem,
                "name": data.get("name") or "(unnamed)",
                "updated_at": data.get("updated_at", ""),
                "turns": sum(1 for m in data.get("messages", []) if m.get("role") == "user"),
            })
        except (json.JSONDecodeError, OSError):
            pass
    return sorted(items, key=lambda x: x["updated_at"], reverse=True)


def _find_chat(spec):
    items = _list_chats()
    if not items:
        return None
    if not spec:
        return items[0]
    by_id = next((i for i in items if i["id"] == spec), None)
    if by_id:
        return by_id
    return next((i for i in items if spec.lower() in i["name"].lower()), None)


def _new_chat():
    return {
        "id": uuid.uuid4().hex[:8],
        "name": None,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "conversation_id": None,
        "parent_message_id": "00000000-0000-0000-0000-000000000000",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _save_chat(chat):
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    (CHATS_DIR / f"{chat['id']}.json").write_text(json.dumps(chat, indent=2))


# --- async stream bridge: sync generator → async iterator ---
async def _stream_to_async(sync_iter):
    """Run a sync generator in a thread, yield its items in async land."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def runner():
        try:
            for chunk in sync_iter:
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except BaseException as e:  # propagate including KeyboardInterrupt
            loop.call_soon_threadsafe(q.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, DONE)

    threading.Thread(target=runner, daemon=True).start()
    while True:
        item = await q.get()
        if item is DONE:
            return
        if isinstance(item, tuple) and item and item[0] == "error":
            raise item[1]
        yield item


# --- agent loop (async, prompt_toolkit-driven) ---
async def main_async():
    from collections import deque

    parser = argparse.ArgumentParser(prog="cli-agent")
    parser.add_argument("-r", "--resume", nargs="?", const="", default=None,
                        metavar="NAME_OR_ID",
                        help="resume a chat (most recent if no name given)")
    parser.add_argument("-l", "--list-sessions", action="store_true",
                        help="list saved chats and exit")
    parser.add_argument("--mode", choices=["safe", "auto-edit", "yolo"],
                        default=os.environ.get("CLI_AGENT_MODE", "safe"),
                        help="permission level (default: safe)")
    parser.add_argument("--yolo", action="store_true", help="alias for --mode yolo")
    parser.add_argument("--reminder-every", type=int,
                        default=int(os.environ.get("CLI_AGENT_REMINDER_EVERY", "0")),
                        metavar="N",
                        help="re-inject tool-use reminder every N user turns (0=off, default 0).")
    args = parser.parse_args()
    if args.yolo:
        args.mode = "yolo"
    mode = args.mode

    if args.list_sessions:
        items = _list_chats()
        if not items:
            print("(no saved chats)"); return
        for c in items:
            print(f"{c['id']}  {c['updated_at'][:19]}  ({c['turns']} turns)  {c['name']}")
        return

    if args.resume is not None:
        match = _find_chat(args.resume)
        if not match:
            sys.exit(f"no matching chat for '{args.resume}'")
        chat = json.loads((CHATS_DIR / f"{match['id']}.json").read_text())
        print(f"resumed chat {chat['id']}: {chat.get('name') or '(unnamed)'}  "
              f"({sum(1 for m in chat['messages'] if m['role'] == 'user')} turns)")
    else:
        chat = _new_chat()

    if LIBRECHAT_URL:
        backend = LibreChatBackend()
        backend.conversation_id = chat["conversation_id"]
        backend.parent_message_id = chat["parent_message_id"]
        backend._first_turn = chat["conversation_id"] is None
        print(f"agent ready (LibreChat={LIBRECHAT_URL}, model={LIBRECHAT_MODEL}, endpoint={LIBRECHAT_ENDPOINT})")
    elif OPENAI_BASE_URL or os.environ.get("OPENAI_API_KEY"):
        backend = OpenAIBackend()
        print(f"agent ready (OpenAI-compat={OPENAI_BASE_URL or 'default'}, model={AGENT_MODEL})")
    else:
        sys.exit("no backend configured — set LIBRECHAT_URL or OPENAI_BASE_URL in .env")
    print(f"chat id: {chat['id']}  mode: {mode}  (Ctrl-C twice to quit, /help for commands)\n")

    history_path = CONFIG_DIR / "input_history"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    pending: "deque[str]" = deque()
    new_input_evt = asyncio.Event()
    busy = [False]
    input_task_holder = [None]  # asyncio.Task

    # custom keybindings (Ctrl+O, Ctrl+B, Ctrl+C, Ctrl+J, arrow-up-recall)
    from prompt_toolkit.key_binding import KeyBindings
    kb = KeyBindings()

    def _msg(event, text):
        """Print a message above the prompt (works while prompt is active)."""
        sys.stdout.write(text + "\n"); sys.stdout.flush()

    @kb.add("c-o")
    def _(event):
        p = _LAST_FG_LOG[0]
        if not p or not p.exists():
            _msg(event, "\n(no foreground log yet)")
            return
        text = p.read_text(errors="replace")
        _msg(event, f"\n--- {p} ---\n{text}--- end log ({len(text.splitlines())} lines) ---")

    @kb.add("c-b")
    def _(event):
        if _CURRENT_FG.get("proc") is None:
            _msg(event, "\n(no foreground command running)")
            return
        _FG_BG_REQUESTED.set()
        _msg(event, "\n[Ctrl+B] requesting backgrounding of current foreground command...")

    @kb.add("c-c")
    def _(event):
        proc = _CURRENT_FG.get("proc")
        if proc is not None:
            try: proc.terminate()
            except Exception: pass
            _msg(event, "\n[Ctrl+C] terminated foreground command")
            return
        # default-ish behavior: clear the input line, or exit on double-tap
        now = time.monotonic()
        if event.current_buffer.text:
            event.current_buffer.reset()
            return
        if now - _last_sigint[0] < 1.5:
            event.app.exit(exception=KeyboardInterrupt)
        _last_sigint[0] = now

    @kb.add("c-j")
    def _(event):
        # push a sentinel that the main loop will resolve to an interactive menu
        pending.append("__JOBS_MENU__")
        new_input_evt.set()
        if not busy[0]:
            event.app.exit(result="")

    @kb.add("up")
    def _(event):
        # if there's a queued message, recall the newest one into the buffer.
        # otherwise fall through to history navigation
        if pending:
            text = pending.pop()
            buf = event.current_buffer
            buf.text = text
            buf.cursor_position = len(text)
        else:
            event.current_buffer.history_backward()

    session = PromptSession(history=FileHistory(str(history_path)), key_bindings=kb)

    async def input_loop():
        """Always running — feeds 'pending' deque. Cancelled briefly during confirm."""
        try:
            while True:
                default = _RECALL_TEXT[0]
                _RECALL_TEXT[0] = ""
                try:
                    line = await session.prompt_async("> ", default=default)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    pending.append(None)  # sentinel: exit
                    new_input_evt.set()
                    return
                line = (line or "").strip()
                if not line:
                    continue
                pending.append(line)
                new_input_evt.set()
                if busy[0]:
                    print(f"  [queued #{len(pending)}: {line[:80]}{'...' if len(line) > 80 else ''}]")
        except asyncio.CancelledError:
            return

    async def get_input():
        while not pending:
            new_input_evt.clear()
            await new_input_evt.wait()
        return pending.popleft()

    async def confirm(name, ar):
        """Pause input_loop, prompt, restart input_loop. Returns 'y' / 'n' / 'a' / 'q'."""
        nonlocal mode
        if name not in _TOOLS_NEEDING_CONFIRM[mode]:
            return "y"
        preview = json.dumps(ar)[:200]
        print(f"\n  [ASK] {name}({preview})")
        if input_task_holder[0] is not None:
            input_task_holder[0].cancel()
            try:
                await input_task_holder[0]
            except asyncio.CancelledError:
                pass
        try:
            ans = (await session.prompt_async(
                "  allow? [y]es / [n]o / [a]ll-from-now / [q]uit: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        finally:
            input_task_holder[0] = asyncio.create_task(input_loop())
        if ans == "a":
            mode = "yolo"; return "y"
        if ans == "q":
            raise SystemExit(0)
        return "y" if ans == "y" else "n"

    messages = chat["messages"]
    input_task_holder[0] = asyncio.create_task(input_loop())

    try:
        with patch_stdout(raw=True):
            while True:
                user = await get_input()
                if user is None:  # EOF sentinel
                    return

                if user == "__JOBS_MENU__":
                    # interactive jobs menu (triggered by Ctrl+J)
                    while True:
                        items = list(_BG_JOBS.items())
                        if not items:
                            print("\n(no background jobs)")
                            break
                        print("\nbackground jobs:")
                        for i, (jid, job) in enumerate(items, 1):
                            rc = job["proc"].poll()
                            status = "running" if rc is None else f"exited (rc={rc})"
                            elapsed = int(time.time() - job["started"])
                            print(f"  {i}. {jid}  {status:<18} {elapsed:>4}s  {job['command'][:60]}")
                        try:
                            ans = await session.prompt_async(
                                "action: [v <n> view, k <n> kill, q quit] > ")
                        except (EOFError, KeyboardInterrupt):
                            break
                        ans = ans.strip()
                        if not ans or ans == "q":
                            break
                        parts = ans.split(maxsplit=1)
                        op = parts[0]
                        if len(parts) < 2:
                            print("  (need a job number, e.g. 'v 1')"); continue
                        try:
                            n = int(parts[1])
                        except ValueError:
                            print("  (number please)"); continue
                        if not 1 <= n <= len(items):
                            print(f"  (out of range — pick 1..{len(items)})"); continue
                        jid, job = items[n - 1]
                        if op == "v":
                            print(f"--- {job['log']} (last 50 lines) ---")
                            print(_tail_file(job["log"], 50))
                        elif op == "k":
                            print(tool_kill_bash(jid))
                        else:
                            print("  (use v or k)")
                    continue

                if user.startswith("/"):
                    cmd, _, rest = user[1:].partition(" ")
                    if cmd in ("exit", "quit"):
                        return
                    if cmd == "help":
                        print("  /name <name>   rename current chat\n"
                              "  /list          list chats\n"
                              "  /id            show this chat's id\n"
                              "  /mode <m>      change permission mode (safe|auto-edit|yolo)\n"
                              "  /jobs          list background jobs\n"
                              "  /kill <id>     kill a background job\n"
                              "  /log [id]      show full log (fg if no id, else bg job id)\n"
                              "  /logs          list all saved logs\n"
                              "  /queue         show queued messages\n"
                              "  /recall        pop newest queued message into prompt for editing\n"
                              "  /exit          quit\n"
                              "keys:\n"
                              "  Ctrl+C   kill running fg command, or clear prompt, or (double-tap) exit\n"
                              "  Ctrl+O   show full log of last foreground command\n"
                              "  Ctrl+B   move running fg command to background (then continue chatting)\n"
                              "  Ctrl+J   open interactive background-jobs menu (view/kill)\n"
                              "  Up arrow recall newest queued msg if any, else browse history\n"
                              "  type while busy → line is queued, auto-sent at next turn boundary")
                        continue
                    if cmd == "jobs":      print(tool_list_bg_jobs()); continue
                    if cmd == "kill":
                        jid = rest.strip()
                        print(tool_kill_bash(jid) if jid else "usage: /kill <job_id>"); continue
                    if cmd == "queue":
                        if not pending:
                            print("(no queued messages)")
                        else:
                            for i, q in enumerate(pending, 1):
                                print(f"  {i}. {q}")
                        continue
                    if cmd == "recall":
                        if not pending:
                            print("(nothing to recall)")
                        else:
                            _RECALL_TEXT[0] = pending.pop()
                            print("recalled — it'll appear in the next prompt to edit")
                        continue
                    if cmd == "log":
                        arg = rest.strip()
                        if not arg:
                            p = _LAST_FG_LOG[0]
                            if not p or not p.exists():
                                print("(no foreground log yet)"); continue
                            print(f"--- {p} ---\n{p.read_text(errors='replace')}")
                        else:
                            candidates = [_BG_JOBS_DIR / f"{arg}.log", _BG_JOBS_DIR / arg, Path(arg)]
                            matches = list(_BG_JOBS_DIR.glob(f"{arg}*.log")) if _BG_JOBS_DIR.exists() else []
                            p = next((c for c in candidates if c.exists()), None) or (matches[0] if matches else None)
                            if not p:
                                print(f"no log found for '{arg}'")
                            else:
                                print(f"--- {p} ---\n{p.read_text(errors='replace')}")
                        continue
                    if cmd == "logs":
                        if not _BG_JOBS_DIR.exists():
                            print("(no logs)"); continue
                        entries = sorted(_BG_JOBS_DIR.glob("*.log"),
                                         key=lambda p: p.stat().st_mtime, reverse=True)
                        if not entries:
                            print("(no logs)"); continue
                        for p in entries[:30]:
                            print(f"  {p.name:<30}  {p.stat().st_size:>8}b  "
                                  f"{int(time.time() - p.stat().st_mtime)}s ago")
                        if len(entries) > 30:
                            print(f"  ... ({len(entries) - 30} more)")
                        continue
                    if cmd == "name":
                        chat["name"] = rest.strip() or chat["name"]
                        _save_chat(chat); print(f"named: {chat['name']}"); continue
                    if cmd == "list":
                        for c in _list_chats():
                            print(f"  {c['id']}  ({c['turns']}t)  {c['name']}")
                        continue
                    if cmd == "id":
                        print(f"  {chat['id']}"); continue
                    if cmd == "mode":
                        m = rest.strip()
                        if m in _TOOLS_NEEDING_CONFIRM:
                            mode = m; print(f"mode: {mode}")
                        else:
                            print(f"current mode: {mode} (options: safe, auto-edit, yolo)")
                        continue
                    print("(unknown command — try /help)"); continue

                if not chat["name"]:
                    chat["name"] = user[:60]
                chat["turn_count"] = chat.get("turn_count", 0) + 1
                messages.append({"role": "user", "content": user})

                send_messages = messages
                if args.reminder_every > 0 and chat["turn_count"] % args.reminder_every == 0:
                    if isinstance(backend, LibreChatBackend):
                        send_messages = list(messages)
                        last = dict(send_messages[-1])
                        last["content"] = f"[{TOOL_REMINDER}]\n\n{last['content']}"
                        send_messages[-1] = last
                    else:
                        send_messages = list(messages)
                        send_messages.insert(-1, {"role": "system", "content": TOOL_REMINDER})

                busy[0] = True
                try:
                    for _ in range(20):
                        assistant_text = ""
                        print()
                        async for delta in _stream_to_async(backend.send_stream(send_messages)):
                            sys.stdout.write(delta); sys.stdout.flush()
                            assistant_text += delta
                        print()
                        messages.append({"role": "assistant", "content": assistant_text})
                        if isinstance(backend, LibreChatBackend):
                            chat["conversation_id"] = backend.conversation_id
                            chat["parent_message_id"] = backend.parent_message_id

                        calls = extract_tool_calls(assistant_text)
                        if not calls:
                            print()
                            break
                        results = []
                        for name, ar in calls:
                            preview = {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                                       for k, v in (ar.items() if isinstance(ar, dict) else [])}
                            if name == "__PARSE_ERROR__":
                                results.append(f"<tool_response>parse error: {ar.get('error')}</tool_response>")
                                continue
                            if name in ("write_file", "edit_file"):
                                _show_diff_for(name, ar)
                            decision = await confirm(name, ar)
                            if decision != "y":
                                print(f"  [skip] {name}")
                                results.append(f"<tool_response>user declined to run {name}</tool_response>")
                                continue
                            print(f"  [tool] {name}({preview})")
                            # tool runs in a thread so we don't block the event loop
                            # (and so KeyboardInterrupt during run_bash works)
                            result = await asyncio.to_thread(call_tool, name, ar)
                            results.append(f"<tool_response>\n{result}\n</tool_response>")
                        messages.append({"role": "user", "content": "\n".join(results)})
                        send_messages = messages
                    else:
                        print("(stopped: tool-call chain exceeded 20 iterations)")
                except KeyboardInterrupt:
                    print("\n(interrupted — back to prompt; press Ctrl-C again quickly to exit)")
                    while messages and messages[-1]["role"] != "user":
                        messages.pop()
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()
                finally:
                    busy[0] = False
                    if pending:
                        print(f"  [{len(pending)} queued message{'s' if len(pending) > 1 else ''} will auto-send]")

                _save_chat(chat)
    finally:
        if input_task_holder[0] is not None:
            input_task_holder[0].cancel()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
