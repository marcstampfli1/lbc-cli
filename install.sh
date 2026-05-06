#!/usr/bin/env bash
# install.sh — installs cli-agent so `cli-agent` works from any directory.
#   - copies sources to $XDG_DATA_HOME/cli-agent (default: ~/.local/share/cli-agent)
#   - puts config at $XDG_CONFIG_HOME/cli-agent (default: ~/.config/cli-agent)
#   - drops a launcher at ~/.local/bin/cli-agent
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/cli-agent"
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cli-agent"
BIN_DIR="$HOME/.local/bin"

echo "==> installing cli-agent"
echo "    source : $SRC"
echo "    app    : $APP_DIR"
echo "    config : $CFG_DIR"
echo "    bin    : $BIN_DIR/cli-agent"
echo

mkdir -p "$APP_DIR" "$CFG_DIR" "$BIN_DIR"

echo "==> copying sources"
cp "$SRC/agent.py" "$SRC/login.py" "$SRC/login_api.py" "$SRC/login_cookie.py" "$SRC/requirements.txt" "$APP_DIR/"

echo "==> creating venv + installing deps"
if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> ensuring playwright chromium is available"
"$APP_DIR/.venv/bin/playwright" install chromium 2>&1 | tail -3 || true

if [ ! -f "$CFG_DIR/.env" ]; then
    cp "$SRC/.env.example" "$CFG_DIR/.env"
    echo "==> wrote default config: $CFG_DIR/.env"
else
    echo "==> kept existing config: $CFG_DIR/.env"
fi

echo "==> writing launcher: $BIN_DIR/cli-agent"
cat > "$BIN_DIR/cli-agent" <<EOF
#!/usr/bin/env bash
# cli-agent launcher — preserves cwd, reads config from $CFG_DIR
export CLI_AGENT_CONFIG_DIR="$CFG_DIR"
case "\${1:-run}" in
    run)           shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/agent.py" "\$@" ;;
    list|ls)       shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/agent.py" --list-sessions "\$@" ;;
    resume)        shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/agent.py" --resume "\${1:-}" ;;
    login)         shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/login.py" "\$@" ;;
    login-api)     shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/login_api.py" "\$@" ;;
    login-cookie)  shift; exec "$APP_DIR/.venv/bin/python" "$APP_DIR/login_cookie.py" "\$@" ;;
    config)        "\${EDITOR:-vi}" "$CFG_DIR/.env" ;;
    where)         echo "app:    $APP_DIR"; echo "config: $CFG_DIR"; echo "bin:    $BIN_DIR/cli-agent" ;;
    -h|help)
        cat <<'USAGE'
usage: cli-agent [command] [flags]
  run            (default) start a new chat in the current directory
  list, ls       list saved chats
  resume [name]  resume a chat by id or name (most recent if no name)
  login          open a real browser to log in (best for SSO/MFA flows)
  login-api      log in via API (prompts for email + password)
  login-cookie   paste a refreshToken cookie copied from your browser DevTools
  config         edit the config file (.env)
  where          print install paths

flags for run/resume:
  --mode {safe|auto-edit|yolo}   permission level (default: safe)
                                 safe       = ask before run_bash and write_file
                                 auto-edit  = ask before run_bash only
                                 yolo       = never ask (USE AT YOUR OWN RISK)
  --yolo                         alias for --mode yolo
  -r, --resume [NAME]            same as 'resume' subcommand

in-chat slash commands: /name <name>, /list, /id, /mode <m>, /help, /exit
controls: Ctrl-C = interrupt agent / clear prompt, Ctrl-C twice quickly = exit
USAGE
        ;;
    --help|-*)     exec "$APP_DIR/.venv/bin/python" "$APP_DIR/agent.py" "\$@" ;;
    *) echo "unknown command: \$1 (try: cli-agent help)"; exit 1 ;;
esac
EOF
chmod +x "$BIN_DIR/cli-agent"

echo
echo "==> done."

case ":$PATH:" in
    *":$BIN_DIR:"*)
        echo "    \$PATH already includes $BIN_DIR — you're set."
        ;;
    *)
        echo "    NOTE: $BIN_DIR is not on your \$PATH."
        echo "    add this line to ~/.bashrc or ~/.zshrc and reopen your shell:"
        echo "        export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
esac

cat <<EOF

next steps:
  1.  cli-agent config        # set LIBRECHAT_URL (and OPENAI_BASE_URL if testing locally)
  2.  cli-agent login         # browser-based auth (or 'cli-agent login-api' for password)
  3.  cd /any/project && cli-agent
EOF
