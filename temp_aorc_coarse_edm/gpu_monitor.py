"""
gpu_monitor.py — background GPU utilisation poller.

Usage:
    monitor = GPUMonitor(gpu_id=0)
    ...
    util, avg_util, mem_used_mib, mem_total_mib = monitor.stats()
    monitor.stop()
"""

import subprocess
import threading
import time
from collections import deque


class GPUMonitor:
    def __init__(self, gpu_id: int = 0, interval: float = 1.0, window: int = 10):
        self.gpu_id = gpu_id
        self.interval = interval
        self._util_window: deque = deque(maxlen=window)
        self._mem_used = 0
        self._mem_total = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_id}",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                parts = result.stdout.strip().split(",")
                util      = int(parts[0].strip())
                mem_used  = int(parts[1].strip())
                mem_total = int(parts[2].strip())
                with self._lock:
                    self._util_window.append(util)
                    self._mem_used  = mem_used
                    self._mem_total = mem_total
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self):
        """Returns (current_util%, avg_util%, mem_used_MiB, mem_total_MiB)."""
        with self._lock:
            utils = list(self._util_window)
            cur   = utils[-1] if utils else -1
            avg   = sum(utils) / len(utils) if utils else -1.0
            return cur, avg, self._mem_used, self._mem_total

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def log_and_warn(self, step: int, low: int = 90, high: int = 95) -> str:
        """
        Returns a formatted GPU status string and prints a hint if utilisation
        is outside [low, high].
        """
        cur, avg, mem_used, mem_total = self.stats()
        mem_pct = 100 * mem_used / mem_total if mem_total else 0
        status = (
            f"GPU util {cur:3d}% (avg {avg:.0f}%) | "
            f"VRAM {mem_used}/{mem_total} MiB ({mem_pct:.0f}%)"
        )
        if avg != -1:
            if avg < low:
                status += f"  [HINT] avg util {avg:.0f}% < {low}% — try increasing batch_size"
            elif avg > high:
                status += f"  [WARN] avg util {avg:.0f}% > {high}% — risk of OOM, reduce batch_size"
        return status
