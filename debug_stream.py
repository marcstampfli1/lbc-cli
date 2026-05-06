"""Two-step: POST /api/agents/chat/ollama then GET /stream/<id>, dump raw events."""
import json, uuid
from datetime import datetime
import httpx
from pathlib import Path

URL = "http://localhost:3080"
state = json.loads(Path("session.json").read_text())
token = state["token"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

payload = {
    "text": "Reply with exactly: hello",
    "sender": "User",
    "clientTimestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    "isCreatedByUser": True,
    "parentMessageId": "00000000-0000-0000-0000-000000000000",
    "messageId": str(uuid.uuid4()),
    "error": False,
    "endpoint": "ollama",
    "endpointType": "openAI",
    "model": "qwen2.5:3b",
    "key": "never",
    "isTemporary": False,
    "isRegenerate": False,
    "isContinued": False,
}

with httpx.Client(base_url=URL, timeout=180.0,
                  headers={"Authorization": f"Bearer {token}", "User-Agent": UA}) as c:
    r = c.post("/api/agents/chat/ollama", json=payload)
    print(f"POST {r.status_code}: {r.text[:300]}")
    init = r.json()
    sid = init["streamId"]
    print(f"streaming /api/agents/chat/stream/{sid}\n")

    with c.stream("GET", f"/api/agents/chat/stream/{sid}",
                  headers={"Accept": "text/event-stream"}) as sr:
        print(f"GET {sr.status_code} content-type={sr.headers.get('content-type')}\n")
        for i, line in enumerate(sr.iter_lines()):
            print(f"{i:3d}| {line!r}")
            if i > 80:
                print("(truncated)"); break
