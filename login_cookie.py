#!/usr/bin/env python3
"""Cookie-paste login — for when you can't run a browser locally but already
have a LibreChat session in some other browser. Prompt for the refreshToken
cookie value, exchange it for a JWT, save session.json."""
import getpass
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

CONFIG_DIR = Path(os.environ.get("CLI_AGENT_CONFIG_DIR") or
                  Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "cli-agent")
_env_file = CONFIG_DIR / ".env"
load_dotenv(_env_file if _env_file.exists() else None)

URL = os.environ.get("LIBRECHAT_URL", "").rstrip("/")
SESSION_FILE = os.environ.get("SESSION_FILE", "session.json")
if not Path(SESSION_FILE).is_absolute():
    SESSION_FILE = str(CONFIG_DIR / SESSION_FILE)

if not URL:
    sys.exit("LIBRECHAT_URL not set — run 'cli-agent config' first")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

print(f"target: {URL}")
print(f"\nIn your browser, open DevTools (F12) on a logged-in LibreChat tab,")
print("go to: Application → Storage → Cookies → <LibreChat domain>")
print("copy the value of the 'refreshToken' cookie and paste below.\n")
refresh_token = getpass.getpass("refreshToken: ").strip()
if not refresh_token:
    sys.exit("no token entered")

with httpx.Client(base_url=URL, timeout=30.0,
                  headers={"User-Agent": UA},
                  cookies={"refreshToken": refresh_token}) as c:
    r = c.post("/api/auth/refresh")
    if not r.ok:
        sys.exit(f"refresh failed (HTTP {r.status_code}): {r.text[:300]}")
    body = r.json()
    token = body["token"]

state = {
    "cookies": [{"name": "refreshToken", "value": refresh_token, "domain": "", "path": "/"}],
    "token": token,
    "origins": [],
}
Path(SESSION_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(SESSION_FILE).write_text(json.dumps(state, indent=2))
print(f"\nsaved session to {SESSION_FILE}")

with httpx.Client(base_url=URL, timeout=30.0,
                  headers={"Authorization": f"Bearer {token}", "User-Agent": UA}) as c:
    print("\n--- /api/endpoints ---")
    print(c.get("/api/endpoints").text[:1500])
