from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


class LocalLauncherTest(unittest.TestCase):
    def test_launcher_is_executable_visible_and_shell_valid(self):
        from sim.local_launcher import build_local_worker_launcher_zip

        payload = build_local_worker_launcher_zip("token with spaces")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("启动本地算力.command", archive.namelist())
            self.assertIn("启动说明.txt", archive.namelist())
            self.assertIn("worker_payload/local_worker_server.py", archive.namelist())
            self.assertIn("worker_payload/sim/bots.py", archive.namelist())
            self.assertIn("worker_payload/sim/ai_training.py", archive.namelist())
            self.assertIn("worker_payload/sim/finance.py", archive.namelist())
            self.assertIn("worker_payload/check_health.py", archive.namelist())
            self.assertIn("worker_payload/start_background.py", archive.namelist())
            info = archive.getinfo("启动本地算力.command")
            self.assertEqual((info.external_attr >> 16) & 0o777, 0o755)
            script = archive.read(info).decode("utf-8")
            instructions = archive.read("启动说明.txt").decode("utf-8")

        result = subprocess.run(
            ["/bin/zsh", "-n"], input=script, text=True,
            capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SUPER_BOT_REMOTE_TOKEN='token with spaces'", script)
        self.assertIn('SUPER_BOT_PARALLEL_WORKERS="${SUPER_BOT_PARALLEL_WORKERS:-8}"', script)
        self.assertIn('exec > >(tee -a "$LOG_FILE") 2>&1', script)
        self.assertIn('osascript - "$PUBLIC_URL"', script)
        self.assertIn("PUBLIC_READY", script)
        self.assertIn("for TUNNEL_ATTEMPT in 1 2 3", script)
        self.assertIn('"$PUBLIC_URL/health"', script)
        self.assertIn("@223.5.5.5", script)
        self.assertIn('--resolve "$PUBLIC_HOST:443:$PUBLIC_IP"', script)
        self.assertIn("grep -av '^https://api\\.'", script)
        self.assertIn('kill -0 "$TUNNEL_PID"', script)
        self.assertIn("无需连接 GitHub", script)
        self.assertNotIn("git clone", script)
        self.assertIn("不需要连接 GitHub", instructions)
        self.assertIn("不会闪退", instructions)
        self.assertIn("已保留原有地址", script)

    def test_health_check_rejects_stale_or_wrong_services(self):
        from sim.local_launcher import build_local_worker_launcher_zip
        from sim.remote_worker import PROTOCOL, calculation_revision

        payload = {"ok": True, "protocol": PROTOCOL, "service": "business-sim-super-bot",
                   "calculation_revision": calculation_revision()}
        with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(io.BytesIO(build_local_worker_launcher_zip("test"))) as archive:
            archive.extractall(temp)
            checker = Path(temp) / "worker_payload/check_health.py"
            for response, rc in ((payload, 0), (dict(payload, calculation_revision={"bot": -1}), 1),
                                 (dict(payload, service="unrelated"), 1), (None, 1)):
                with self.subTest(response=response):
                    result = subprocess.run([sys.executable, str(checker)], input=json.dumps(response),
                                            text=True, capture_output=True, check=False)
                    self.assertEqual(result.returncode, rc, result.stderr)


if __name__ == "__main__":
    unittest.main()
