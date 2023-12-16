import subprocess

import psutil


def start_tensorboard(logdir: str, port: int = 6066):
    host = "127.0.0.1"
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port:
                pid = conn.pid
                if pid != 0:
                    psutil.Process(pid).terminate()
    except NameError:
        ...
    finally:
        cmd = ["tensorboard", "--logdir", str(logdir), "--port", str(port), "--host", host]
        with open(".tb_log.txt", "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=f, shell=False)
        print(f"Tensorboard has started at http://{host}:{port}")
