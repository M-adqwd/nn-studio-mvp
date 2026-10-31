"""ModelForge 启动器：一条命令拉起后端 API 与前端静态服务。

为什么不用 start.ps1：
  PowerShell 5.1 的 Start-Process 会因为环境里同时存在 HTTPS_PROXY 和
  https_proxy（大小写不同的重复键）直接抛 ArgumentException，
  而国内网络环境里这两种写法经常同时出现。用 Python 拉子进程不受此影响。

用法：
  python start.py          启动（已在运行则先停旧的）
  python start.py --stop   停止两个服务
"""
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PORTS = (8000, 8096)


def _port_pids(port: int) -> list[int]:
    """找出监听指定端口的进程 PID（Windows 用 netstat，跨平台回退 psutil-free 方案）。"""
    pids: list[int] = []
    if os.name == "nt":
        try:
            # netstat on a Chinese Windows emits GBK bytes; decoding them as
            # UTF-8 raises UnicodeDecodeError inside the reader thread.
            r = subprocess.run(["netstat", "-ano"], capture_output=True,
                               timeout=20)
            out = (r.stdout or b"").decode("gbk", errors="ignore")
        except Exception:
            return pids
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[-1].isdigit():
                if "LISTENING" in line.upper():
                    pids.append(int(parts[-1]))
    return pids


def stop_all() -> None:
    for port in PORTS:
        for pid in _port_pids(port):
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                   capture_output=True, timeout=20)
                else:
                    os.kill(pid, signal.SIGTERM)
                print(f"  stopped port {port} (pid {pid})")
            except Exception as e:  # noqa: BLE001
                print(f"  failed to stop port {port}: {e}")
    time.sleep(1.5)


def spawn(args: list[str], cwd: Path, log: Path) -> subprocess.Popen:
    """拉起子进程。日志重定向到文件，脱离当前控制台。

    关键：必须用 DETACHED_PROCESS。
    只加 CREATE_NEW_PROCESS_GROUP 的话，父进程退出时子进程仍会被
    连带杀掉（进程组被一起清理），服务刚启动就没了。
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(log, "ab") as fh:
        return subprocess.Popen(
            args, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, close_fds=True, creationflags=flags)


def wait_ready(url: str, timeout: int = 45) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def main() -> None:
    if "--stop" in sys.argv:
        print("stopping ModelForge ...")
        stop_all()
        print("done")
        return

    print("stopping old instances ...")
    stop_all()

    print("starting backend  -> http://127.0.0.1:8000")
    spawn([PY, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
           "--port", "8000", "--log-level", "warning"],
          ROOT / "backend", ROOT / "logs" / "api.out")

    print("starting frontend -> http://127.0.0.1:8096/index.html")
    spawn([PY, "-m", "http.server", "8096", "--bind", "127.0.0.1"],
          ROOT / "frontend", ROOT / "logs" / "web.out")

    ok_api = wait_ready("http://127.0.0.1:8000/api/health")
    ok_web = wait_ready("http://127.0.0.1:8096/index.html")

    print()
    if ok_api and ok_web:
        print("READY")
        print("  UI  : http://127.0.0.1:8096/index.html")
        print("  User: admin / Admin@123")
    else:
        print(f"backend ready={ok_api}  frontend ready={ok_web}")
        print("check logs/api.out and logs/web.out")
        sys.exit(1)


if __name__ == "__main__":
    main()
