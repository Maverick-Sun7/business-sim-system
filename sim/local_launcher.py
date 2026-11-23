"""Build the admin-only macOS local compute launcher."""
from __future__ import annotations

import io
from pathlib import Path
import shlex
import zipfile


LAUNCHER_API_VERSION = 3
_HEALTH_CHECK = '''import json
import sys
from sim.remote_worker import PROTOCOL, calculation_revision

expected = calculation_revision()
if sys.argv[1:] == ["--version"]:
    print(f"协议 {PROTOCOL} / Bot {expected['bot']} / CPI {expected['cpi']} / Engine {expected['engine']}")
    sys.exit(0)
try:
    result = json.load(sys.stdin)
except (ValueError, UnicodeError):
    sys.exit(1)
sys.exit(0 if isinstance(result, dict) and result.get("ok") is True
         and result.get("service") == "business-sim-super-bot"
         and result.get("protocol") == PROTOCOL
         and result.get("calculation_revision") == expected else 1)
'''
_START_BACKGROUND = '''import subprocess
import sys

log_path, *command = sys.argv[1:]
with open(log_path, "ab") as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
print(process.pid)
'''


_WORKER_FILES = (
    "local_worker_server.py",
    "scf_worker.py",
    "sim/__init__.py",
    "sim/ai_training.py",
    "sim/bot_joint_pricing.py",
    "sim/bot_market_forecast.py",
    "sim/bots.py",
    "sim/cpi.py",
    "sim/db.py",
    "sim/defaults.py",
    "sim/engine.py",
    "sim/finance.py",
    "sim/remote_worker.py",
)


def build_local_worker_launcher_zip(token: str) -> bytes:
    """Return a visible, self-diagnosing macOS launcher archive."""
    script = r'''#!/bin/zsh
set -eu
setopt PIPE_FAIL

export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
WORKER_DIR="$HOME/.business-sim-local-worker"
RUNTIME_DIR="$HOME/.business-sim-local-worker-runtime"
SCRIPT_DIR="${0:A:h}"
PAYLOAD_DIR="$SCRIPT_DIR/worker_payload"
LOG_FILE="$RUNTIME_DIR/launcher.log"
CHECK_URL="$PAYLOAD_DIR/check_health.py"
START_BACKGROUND="$PAYLOAD_DIR/start_background.py"
umask 077
mkdir -p "$RUNTIME_DIR"
: >"$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

show_failure() {
  local STATUS="${1:-1}"
  trap - ZERR INT TERM
  echo
  echo "启动失败。完整日志：$LOG_FILE"
  if [[ "${LOCAL_WORKER_NO_DIALOG:-0}" != "1" ]]; then
    /usr/bin/open -a TextEdit "$LOG_FILE" >/dev/null 2>&1 || true
    /usr/bin/osascript - "$LOG_FILE" <<'APPLESCRIPT' || true
on run argv
  display dialog ("本地算力启动失败。\n\n详细日志已经打开：\n" & item 1 of argv) buttons {"好"} default button 1 with icon stop
end run
APPLESCRIPT
    echo "按回车关闭此窗口。"
    read -r || true
  fi
  exit "$STATUS"
}
trap 'show_failure $?' ZERR
trap 'show_failure 130' INT TERM

echo "========================================"
echo "  商赛系统 · 本地算力启动器"
echo "========================================"
echo "[1/5] 检查运行环境"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "没有找到 Python 3。请先安装 Python 3 后重新运行。"
  false
fi
if [[ ! -f "$PAYLOAD_DIR/local_worker_server.py" || ! -f "$CHECK_URL" || ! -f "$START_BACKGROUND" ]]; then
  echo "启动器文件不完整。请完整解压 ZIP 后再双击，不能只拖出 .command 文件。"
  false
fi

echo "[2/5] 检查随包附带的计算代码（无需连接 GitHub）"
EXPECTED_HEALTH="$("$PYTHON_BIN" -B "$CHECK_URL" --version)"
echo "  $EXPECTED_HEALTH"

managed_pid() {
  local KIND="$1" EXPECTED_COMMAND="$2" PID COMMAND
  [[ -r "$RUNTIME_DIR/$KIND.pid" ]] || return 1
  PID="$(<"$RUNTIME_DIR/$KIND.pid")"
  [[ "$PID" == <-> && "$PID" -gt 1 ]] || return 1
  COMMAND="$(/bin/ps -p "$PID" -o command= 2>/dev/null)" || return 1
  [[ "$COMMAND" == *"$EXPECTED_COMMAND"* ]] || return 1
  print -r -- "$PID"
}

health_matches() {
  print -r -- "$1" | "$PYTHON_BIN" -B "$CHECK_URL"
}

echo "[3/5] 检查公网连接工具"
CLOUDFLARED_BIN="$(command -v cloudflared || true)"
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  BREW_BIN="$(command -v brew || true)"
  if [[ -z "$BREW_BIN" ]]; then
    echo "没有找到 cloudflared 或 Homebrew。"
    echo "请先从 https://brew.sh 安装 Homebrew，再重新双击启动器。"
    false
  fi
  echo "首次运行需要安装 cloudflared，请稍候……"
  "$BREW_BIN" install cloudflared
  CLOUDFLARED_BIN="$(command -v cloudflared || true)"
fi
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  echo "cloudflared 安装完成后仍无法找到。"
  false
fi

echo "[4/5] 启动本地计算服务"
OLD_PID="$(managed_pid worker "$WORKER_DIR/local_worker_server.py" || true)"
if [[ -n "$OLD_PID" ]]; then
  if /usr/sbin/lsof -anP -p "$OLD_PID" -iTCP:8765 -sTCP:ESTABLISHED >/dev/null 2>&1; then
    echo "旧计算服务还有活跃请求，为避免中断，请等本轮分析完成后再更新。"
    false
  fi
fi
STAGED_WORKER="$(mktemp -d "$RUNTIME_DIR/worker-code.XXXXXX")"
/usr/bin/ditto "$PAYLOAD_DIR" "$STAGED_WORKER"
if [[ -n "$OLD_PID" ]]; then
  /bin/kill "$OLD_PID"
  for STOP_ATTEMPT in {1..20}; do
    if ! /bin/kill -0 "$OLD_PID" 2>/dev/null; then break; fi
    /bin/sleep 0.25
  done
  if /bin/kill -0 "$OLD_PID" 2>/dev/null; then
    echo "旧计算服务未停止，请稍后重试。"
    false
  fi
fi
if [[ -e "$WORKER_DIR" ]]; then
  BACKUP_DIR="$(mktemp -d "$RUNTIME_DIR/worker-code-previous.XXXXXX")"
  /bin/mv "$WORKER_DIR" "$BACKUP_DIR/worker"
fi
/bin/mv "$STAGED_WORKER" "$WORKER_DIR"

export SUPER_BOT_REMOTE_TOKEN=__SUPER_BOT_REMOTE_TOKEN__
export SUPER_BOT_LOCAL_PORT=8765
export SUPER_BOT_PARALLEL_WORKERS="${SUPER_BOT_PARALLEL_WORKERS:-8}"
: >"$RUNTIME_DIR/worker.log"
WORKER_PID="$("$PYTHON_BIN" -B "$START_BACKGROUND" "$RUNTIME_DIR/worker.log" "$PYTHON_BIN" "$WORKER_DIR/local_worker_server.py")"
print -r -- "$WORKER_PID" >"$RUNTIME_DIR/worker.pid"

WORKER_READY=0
for ATTEMPT in {1..30}; do
  LOCAL_HEALTH="$(/usr/bin/curl -fsS --noproxy '*' --max-time 1 http://127.0.0.1:8765/health 2>/dev/null || true)"
  if health_matches "$LOCAL_HEALTH" && /bin/kill -0 "$WORKER_PID" 2>/dev/null; then
    WORKER_READY=1
    break
  fi
  /bin/sleep 0.5
done
if [[ "$WORKER_READY" != "1" ]]; then
  echo "本地计算服务未返回本次启动对应的规则版本。请检查端口占用或运行日志。"
  tail -80 "$RUNTIME_DIR/worker.log" 2>/dev/null || true
  false
fi

echo "[5/5] 建立 HTTPS 公网线路"
PUBLIC_URL=""
PUBLIC_READY=0
if TUNNEL_PID="$(managed_pid tunnel 'cloudflared tunnel')" && [[ -r "$RUNTIME_DIR/current-url.txt" ]] && [[ "${1:-}" != "--new-line" ]]; then
  PUBLIC_URL="$(<"$RUNTIME_DIR/current-url.txt")"
  if [[ "$PUBLIC_URL" =~ '^https://[-a-z0-9]+\.trycloudflare\.com$' ]]; then
    echo "  检查原有公网地址，尽量让网页继续使用同一个地址……"
    for HEALTH_ATTEMPT in {1..3}; do
      HEALTH_RESPONSE="$(/usr/bin/curl -fs --max-time 5 "$PUBLIC_URL/health" 2>/dev/null || true)"
      if health_matches "$HEALTH_RESPONSE"; then
        PUBLIC_READY=1
        echo "  已保留原有地址，不必重新连接。"
        break
      fi
      /bin/sleep 1
    done
    if [[ "$PUBLIC_READY" != "1" ]] && /bin/kill -0 "$TUNNEL_PID" 2>/dev/null; then
      echo "  本机规则已更新，但原有公网线路未通过验证。已保留此地址；请检查网络或用 --new-line 显式换线。"
      false
    fi
  fi
fi
for TUNNEL_ATTEMPT in 1 2 3; do
  if [[ "$PUBLIC_READY" == "1" ]]; then break; fi
  echo "  正在创建线路（第 $TUNNEL_ATTEMPT/3 次）……"
  if OLD_TUNNEL_PID="$(managed_pid tunnel 'cloudflared tunnel')"; then
    /bin/kill "$OLD_TUNNEL_PID" 2>/dev/null || true
    /bin/sleep 0.5
  fi
  if [[ -s "$RUNTIME_DIR/tunnel.log" ]]; then
    /bin/cp "$RUNTIME_DIR/tunnel.log" "$RUNTIME_DIR/tunnel.previous.log"
  fi
  : >"$RUNTIME_DIR/tunnel.log"
  TUNNEL_PID="$("$PYTHON_BIN" -B "$START_BACKGROUND" "$RUNTIME_DIR/tunnel.log" "$CLOUDFLARED_BIN" tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8765)"
  echo "$TUNNEL_PID" >"$RUNTIME_DIR/tunnel.pid"

  PUBLIC_URL=""
  for URL_ATTEMPT in {1..90}; do
    PUBLIC_URL="$(grep -aEo 'https://[-a-z0-9]+\.trycloudflare\.com' "$RUNTIME_DIR/tunnel.log" | grep -av '^https://api\.' | tail -1 || true)"
    if [[ -n "$PUBLIC_URL" ]]; then break; fi
    if ! /bin/kill -0 "$TUNNEL_PID" 2>/dev/null; then break; fi
    /bin/sleep 0.5
  done
  if [[ -z "$PUBLIC_URL" ]]; then
    echo "  本次没有获得公网地址，自动换线。"
    tail -8 "$RUNTIME_DIR/tunnel.log" 2>/dev/null || true
    continue
  fi

  echo "  已获得地址，正在验证公网可用性：$PUBLIC_URL"
  PUBLIC_HOST="${PUBLIC_URL#https://}"
  for HEALTH_ATTEMPT in {1..45}; do
    if ! /bin/kill -0 "$TUNNEL_PID" 2>/dev/null; then break; fi
    HEALTH_RESPONSE="$(/usr/bin/curl -fs --max-time 3 "$PUBLIC_URL/health" 2>/dev/null || true)"
    if [[ -z "$HEALTH_RESPONSE" ]] && command -v dig >/dev/null 2>&1; then
      # Some mainland DNS resolvers cache a negative answer for a brand-new
      # Quick Tunnel. AliDNS normally sees it immediately, so use that IP only
      # for this local verification while keeping the normal HTTPS hostname.
      PUBLIC_IP="$(dig @223.5.5.5 +time=2 +tries=1 +short "$PUBLIC_HOST" 2>/dev/null | grep -aE '^[0-9]+(\.[0-9]+){3}$' | head -1 || true)"
      if [[ -n "$PUBLIC_IP" ]]; then
        HEALTH_RESPONSE="$(/usr/bin/curl -fs --max-time 5 --resolve "$PUBLIC_HOST:443:$PUBLIC_IP" "$PUBLIC_URL/health" 2>/dev/null || true)"
      fi
    fi
    if health_matches "$HEALTH_RESPONSE"; then
      PUBLIC_READY=1
      break
    fi
    /bin/sleep 1
  done
  if [[ "$PUBLIC_READY" == "1" ]]; then
    break
  fi
  echo "  本次地址无法从公网访问，自动换线。"
done
if [[ "$PUBLIC_READY" != "1" ]]; then
  echo "三次公网线路均未通过验证。请检查网络后重新双击启动器。"
  tail -100 "$RUNTIME_DIR/tunnel.log" 2>/dev/null || true
  false
fi

print -r -- "$PUBLIC_URL" >"$RUNTIME_DIR/current-url.txt"
if [[ "${LOCAL_WORKER_NO_CLIPBOARD:-0}" != "1" ]]; then
  print -rn -- "$PUBLIC_URL" | /usr/bin/pbcopy
fi

echo
echo "========================================"
echo "本地算力启动成功"
echo "$PUBLIC_URL"
if [[ "${LOCAL_WORKER_NO_CLIPBOARD:-0}" != "1" ]]; then
  echo "地址已经复制到剪贴板。"
fi
echo "只有地址变了，才需要回到管理员‘回合控制’，重新连接。"
echo "========================================"

if [[ "${LOCAL_WORKER_NO_DIALOG:-0}" != "1" ]]; then
  /usr/bin/osascript - "$PUBLIC_URL" <<'APPLESCRIPT'
on run argv
  display dialog ("本地算力已启动，HTTPS 地址：\n\n" & item 1 of argv & "\n\n只有地址变了，才需要回到管理员的‘回合控制’页面重新连接。") buttons {"好"} default button 1
end run
APPLESCRIPT

  echo "按回车关闭此窗口；后台算力会继续运行。"
  read -r || true
fi
trap - ZERR INT TERM
'''.replace("__SUPER_BOT_REMOTE_TOKEN__", shlex.quote(str(token)))

    instructions = """商赛系统本地算力（Mac）\n\n1. 完整解压 ZIP，保留启动文件和 worker_payload 文件夹。\n2. 双击“启动本地算力.command”。\n3. 等待窗口显示五步进度。成功后 HTTPS 地址会自动复制；原地址通过验证时会继续使用。\n4. 如果地址变了，回到管理员 → 回合控制，粘贴地址并点击“连接本地算力”。\n\n计算代码已包含在下载包中，不需要连接 GitHub。ZIP 含管理员算力连接凭证，请勿转发。\n如果 macOS 阻止打开，请右键启动文件并选择“打开”。\n发生错误时启动器不会闪退，会自动打开日志并停留在错误画面。\n原地址持续不可用时，可在终端用启动文件的路径加 --new-line 重新建线；新地址须重新连接。\n"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        launcher = zipfile.ZipInfo("启动本地算力.command")
        launcher.create_system = 3
        launcher.external_attr = 0o755 << 16
        launcher.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(launcher, script.encode("utf-8"))

        readme = zipfile.ZipInfo("启动说明.txt")
        readme.create_system = 3
        readme.external_attr = 0o644 << 16
        readme.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(readme, instructions.encode("utf-8"))

        archive.writestr("worker_payload/check_health.py", _HEALTH_CHECK)
        archive.writestr("worker_payload/start_background.py", _START_BACKGROUND)

        project_root = Path(__file__).resolve().parent.parent
        for relative_name in _WORKER_FILES:
            source = project_root / relative_name
            info = zipfile.ZipInfo(f"worker_payload/{relative_name}")
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()
