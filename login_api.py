#!/usr/bin/env python3
"""Programmatic LibreChat login — for testing or when you have the password.
For real chatouille (SSO/manual flow) use login.py instead.

Reads LIBRECHAT_URL from .env, prompts for email/password, saves session.json."""
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
    sys.exit("LIBRECHAT_URL not set in .env")

email = os.environ.get("LIBRECHAT_EMAIL") or input("email: ")
password = os.environ.get("LIBRECHAT_PASSWORD") or getpass.getpass("password: ")

from urllib.parse import urlparse
host = urlparse(URL).hostname or "localhost"

with httpx.Client(base_url=URL, timeout=30.0, verify=False) as c:
    r = c.post("/api/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        sys.exit(f"login failed: HTTP {r.status_code} — {r.text[:300]}")
    try:
        body = r.json()
        token = body["token"]
    except (json.JSONDecodeError, ValueError, KeyError):
        sys.exit(f"login returned unexpected body: {r.text[:300]}")
    cookies = [
        {"name": k, "value": v, "domain": host, "path": "/"}
        for k, v in c.cookies.items()
    ]

state = {"cookies": cookies, "token": token, "origins": []}
Path(SESSION_FILE).write_text(json.dumps(state, indent=2))
print(f"saved session to {SESSION_FILE}")

# probe endpoints/models and suggest .env values
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
with httpx.Client(base_url=URL, timeout=30.0, verify=False,
                  headers={"Authorization": f"Bearer {token}", "User-Agent": UA}) as c:
    ep_r = c.get("/api/endpoints")
    md_r = c.get("/api/models")
    try:
        endpoints = ep_r.json() if ep_r.status_code == 200 else {}
        models = md_r.json() if md_r.status_code == 200 else {}
    except (json.JSONDecodeError, ValueError):
        endpoints, models = {}, {}

print("\n--- /api/endpoints ---")
print(json.dumps(endpoints, indent=2)[:2000])
print("\n--- suggested .env values per endpoint ---")
for name, cfg in endpoints.items():
    is_custom = cfg.get("type") == "custom"
    available_models = models.get(name, [])
    print(f"\n# {name}{' (custom)' if is_custom else ''}")
    print(f"LIBRECHAT_ENDPOINT={name}")
    print(f"LIBRECHAT_ENDPOINT_TYPE={'openAI' if is_custom else ''}")
    if available_models:
        print(f"LIBRECHAT_MODEL={available_models[0]}    # available: {available_models[:5]}")
    else:
        print("LIBRECHAT_MODEL=<not advertised — paste one from the LibreChat UI dropdown>")
