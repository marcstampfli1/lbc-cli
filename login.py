#!/usr/bin/env python3
"""Browser-based login for LibreChat — opens a real Chromium window so the user
can complete any auth flow manually (SSO, MFA, captcha, etc.). After the user
hits Enter, exchanges the resulting refreshToken cookie for a JWT and saves
both to session.json."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

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


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--ignore-certificate-errors"])
        ctx = browser.new_context(user_agent=UA, ignore_https_errors=True)
        page = ctx.new_page()
        print(f"opening {URL} — log in normally in the browser window.")
        page.goto(URL)
        input("\n>>> press ENTER here AFTER you finish logging in <<<\n")

        # exchange refreshToken cookie for a JWT (the access token the agent will use)
        token = None
        try:
            r = ctx.request.post(f"{URL}/api/auth/refresh")
            if r.ok:
                token = r.json().get("token")
        except Exception as e:
            print(f"refresh probe failed: {e}")

        state = ctx.storage_state()
        if token:
            state["token"] = token
        Path(SESSION_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(SESSION_FILE).write_text(json.dumps(state, indent=2))
        print(f"\nsaved session to {SESSION_FILE}")
        if not token:
            print("WARNING: no JWT obtained. The /api/auth/refresh call failed — your session\n"
                  "         cookies were saved but the agent may need a fresh login.")

        # probe endpoints/models so user can fill in .env
        try:
            ep = ctx.request.get(f"{URL}/api/endpoints")
            if ep.ok:
                endpoints = ep.json()
                models_resp = ctx.request.get(f"{URL}/api/models")
                models = models_resp.json() if models_resp.ok else {}
                print("\n--- /api/endpoints ---")
                print(json.dumps(endpoints, indent=2)[:2000])
                print("\n--- suggested .env values per endpoint ---")
                for name, cfg in endpoints.items():
                    is_custom = cfg.get("type") == "custom"
                    avail = models.get(name, [])
                    print(f"\n# {name}{' (custom)' if is_custom else ''}")
                    print(f"LIBRECHAT_ENDPOINT={name}")
                    print(f"LIBRECHAT_ENDPOINT_TYPE={'openAI' if is_custom else ''}")
                    if avail:
                        print(f"LIBRECHAT_MODEL={avail[0]}    # available: {avail[:5]}")
                    else:
                        print("LIBRECHAT_MODEL=<paste from LibreChat dropdown>")
        except Exception as e:
            print(f"(endpoint probe failed: {e})")

        browser.close()


if __name__ == "__main__":
    main()
