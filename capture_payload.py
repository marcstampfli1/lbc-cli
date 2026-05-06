"""Drive the LibreChat UI via Playwright, send one message, dump the request payload
that the frontend produces. Used once to figure out the exact /api/agents/chat shape."""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:3080"
EMAIL = "test@test.test"
PASSWORD = "Testing123!"

captured = []

def on_request(req):
    if "/api/agents/chat" in req.url and req.method == "POST":
        try:
            body = req.post_data
            captured.append({"url": req.url, "headers": dict(req.headers), "body": body})
            print(f"\n=== captured POST {req.url} ===")
            print(body)
        except Exception as e:
            print(f"capture err: {e}")

def main():
    # get refreshToken via API first, then prime the browser's cookies so it auto-auths
    import httpx
    with httpx.Client(base_url=URL) as c:
        c.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD}).raise_for_status()
        cookies = [{"name": k, "value": v, "domain": "localhost", "path": "/", "httpOnly": True}
                   for k, v in c.cookies.items()]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state={"cookies": cookies, "origins": []})
        page = ctx.new_page()
        page.on("request", on_request)
        page.on("requestfailed", lambda r: print(f"requestfailed: {r.url} {r.failure}"))

        page.goto(f"{URL}/c/new", wait_until="domcontentloaded")
        print(f"current url: {page.url}")
        try:
            page.wait_for_selector("textarea", timeout=15000)
        except Exception as e:
            page.screenshot(path="debug_no_textarea.png")
            print(f"no textarea: {e}")
            print(f"final url: {page.url}")
            sys.exit(1)

        # pick the Ollama model — open the endpoint menu
        # try several selector strategies since LibreChat UI varies
        try:
            page.wait_for_selector("textarea", timeout=10000)
            ta = page.locator("textarea").first
            ta.click()
            ta.fill("hello, what is 2+2?")
            page.keyboard.press("Enter")
        except Exception as e:
            print(f"failed to send: {e}")

        # give it a moment for the request to fire
        page.wait_for_timeout(3000)
        browser.close()

    if captured:
        Path("captured_payload.json").write_text(json.dumps(captured, indent=2))
        print(f"\n=== saved {len(captured)} captures to captured_payload.json ===")
    else:
        print("no requests captured")
        sys.exit(1)

if __name__ == "__main__":
    main()
