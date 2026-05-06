# cli-agent

A small CLI coding agent that talks to a LibreChat-hosted LLM (e.g. chatouille / Qwen)
through LibreChat's webui REST API. It can `read_file`, `write_file`, `list_dir`, and
`run_bash` in whatever directory you launch it from.

The agent does **not** call any LLM provider directly. Every model call goes
`agent → LibreChat (your org's webui API) → whatever endpoint LibreChat is configured to use`.

## Install

From this directory:

```sh
./install.sh
```

This will:
- copy the source to `~/.local/share/cli-agent/`
- create a venv there with `openai`, `httpx`, `python-dotenv`, `playwright`
- ensure Playwright's Chromium build is downloaded (open source, BSD-3-Clause)
- write a default config to `~/.config/cli-agent/.env`
- install a launcher at `~/.local/bin/cli-agent`

If `~/.local/bin` is not on your `$PATH`, the installer prints a one-liner to add.

## Configure

```sh
cli-agent config
```

Set at minimum:

```
LIBRECHAT_URL=https://chatouille.example.intra
```

Other fields (`LIBRECHAT_MODEL`, `LIBRECHAT_ENDPOINT`, `LIBRECHAT_ENDPOINT_TYPE`)
can be left blank — running `cli-agent login` (or one of the other login modes)
will print suggested values it discovered from `/api/endpoints`.

## Logging in

Three modes — pick whichever fits your situation.

### 1. Browser window (recommended for SSO)

```sh
cli-agent login
```

A real Chromium window opens. Log in normally — go through SSO redirects, MFA,
captchas, anything your org needs. Come back to the terminal and press Enter.
Cookies + a freshly-minted JWT are saved to `~/.config/cli-agent/session.json`.

This works for any auth flow your browser can handle.

### 2. Email + password (no browser)

```sh
cli-agent login-api
```

Prompts for email and password and POSTs to `/api/auth/login`. Only works
if your LibreChat instance allows password login (chatouille usually does
not — it's SSO-gated).

### 3. Paste cookies (manual, no automation)

```sh
cli-agent login-cookie
```

Use this when you already have a browser session somewhere else and just want
to copy the auth into the CLI. The script asks you to paste the value of the
`refreshToken` cookie. It then exchanges that cookie for a JWT.

#### How to find the `refreshToken` cookie

1. In your browser, log into LibreChat normally and leave the tab open.
2. Open DevTools (F12 / right-click → Inspect).
3. Go to **Application** tab (Chrome/Edge) or **Storage** tab (Firefox).
4. Expand **Cookies** → click the LibreChat domain (e.g. `https://chatouille.intra`).
5. Find the row named **`refreshToken`** and copy its **Value** column.
   It's a long string starting with `eyJ` (a JWT).
6. Paste it into the `cli-agent login-cookie` prompt.

The `refreshToken` is a long-lived (typically 7 days) HttpOnly cookie. It's the
same one your browser uses to keep you logged in. The shorter-lived access JWT
is derived from it.

## Run

```sh
cd /any/project
cli-agent                       # new chat, default safe mode
cli-agent --mode auto-edit      # auto-allow file edits, ask before shell
cli-agent --yolo                # never ask (use carefully)
cli-agent list                  # list saved chats
cli-agent resume                # resume the most recent chat
cli-agent resume my-chat-name   # resume by name (substring match) or id
```

Type a request at the `>` prompt. The agent works in your **current directory** —
all relative paths in tool calls are relative to wherever you `cd`'d before
invoking it.

### Permission modes

| Mode        | Asks before `run_bash` | Asks before `write_file` |
|-------------|:----------------------:|:------------------------:|
| `safe` (default) | yes | yes |
| `auto-edit` | yes | no  |
| `yolo`      | no  | no  |

When the agent asks, you can answer `y` (allow once), `n` (decline), `a`
(allow-all-from-now-on — switches the rest of the session to yolo), or
`q` (quit). `read_file` and `list_dir` are always allowed.

Switch modes mid-chat with `/mode auto-edit` (or `safe`/`yolo`).

### Streaming

The agent's response streams to your terminal token-by-token as the model
generates it. Tool calls show up as raw JSON briefly, then run.

### Controls

- **Ctrl-C once**: interrupts the agent's current generation or tool execution and returns you to the prompt.
- **Ctrl-C twice quickly** (within ~1.5s): exits the agent.
- **Ctrl-D** at the prompt: exits cleanly.
- **`/exit`** or **`/quit`**: exits cleanly.

### In-chat slash commands

```
/name <name>   rename the current chat
/list          list saved chats
/id            show this chat's id
/mode <m>      change permission mode (safe | auto-edit | yolo)
/help          help
/exit          quit
```

Example session:

```
> create a python script hello.py that prints "hi"
  [tool] write_file({'path': 'hello.py', 'content': 'print("hi")'})

I created hello.py.

> run it
  [tool] run_bash({'command': 'python3 hello.py'})

Output: hi
```

## Layout

| What            | Where                              |
|-----------------|------------------------------------|
| code            | `~/.local/share/cli-agent/`        |
| venv            | `~/.local/share/cli-agent/.venv/`  |
| config          | `~/.config/cli-agent/.env`         |
| session         | `~/.config/cli-agent/session.json` |
| launcher        | `~/.local/bin/cli-agent`           |

## How auth works under the hood

LibreChat's API requires three things on every request:

1. A real-looking browser **User-Agent** (anti-bot middleware rejects others).
2. **`Authorization: Bearer <jwt>`** — short-lived access token (~15 min).
3. The **`refreshToken`** cookie — long-lived, lets us mint new JWTs when the access token expires.

`session.json` carries both the cookies and the most recent JWT. The agent will
auto-refresh the JWT on a 401 by POSTing to `/api/auth/refresh`. If the
refreshToken itself expires, log in again.

## Uninstall

```sh
rm -rf ~/.local/share/cli-agent ~/.config/cli-agent ~/.local/bin/cli-agent
```

## Troubleshooting

- **"Illegal request"** from LibreChat — your User-Agent isn't browser-like.
  The agent sets one automatically; if you've hacked the code, restore it.
- **401 Unauthorized** repeatedly — your `refreshToken` is expired. Re-run a login command.
- **"Error parsing conversation"** — `LIBRECHAT_ENDPOINT_TYPE` is wrong for your endpoint.
  Run `cli-agent login` (any mode) and paste a value it suggests, or try `openAI`
  for any custom (OpenAI-compatible) endpoint.
- **Empty replies / "Connection error"** — LibreChat can't reach its configured backend.
  This is server-side, not something the CLI can fix.
