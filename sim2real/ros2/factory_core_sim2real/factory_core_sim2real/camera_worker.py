from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Go2 front camera frames into a local JPEG cache.")
    parser.add_argument("--camera-interface", default="eno1")
    parser.add_argument("--camera-cache-path", default="/tmp/factory_sim2real/front_camera.jpg")
    parser.add_argument("--camera-poll-hz", type=float, default=8.0)
    parser.add_argument("--camera-timeout-sec", type=float, default=3.0)
    args, _unknown = parser.parse_known_args()
    return args


def atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    running = True

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        ChannelFactoryInitialize(0, args.camera_interface or None)
        client = VideoClient()
        client.SetTimeout(float(args.camera_timeout_sec))
        client.Init()
    except Exception as exc:
        print(f"[camera-worker] init failed: {exc}", file=sys.stderr, flush=True)
        return 1

    print(
        f"[camera-worker] capturing on iface={args.camera_interface or 'auto'} "
        f"to {args.camera_cache_path} at {max(args.camera_poll_hz, 0.5):.1f} Hz",
        flush=True,
    )

    poll_period = 1.0 / max(args.camera_poll_hz, 0.5)
    while running:
        started = time.monotonic()
        try:
            code, data = client.GetImageSample()
            if code == 0 and data:
                atomic_write_bytes(args.camera_cache_path, bytes(data))
            else:
                print(f"[camera-worker] non-zero sample code: {code}", file=sys.stderr, flush=True)
                time.sleep(0.2)
        except Exception as exc:
            print(f"[camera-worker] read failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(0.5)

        elapsed = time.monotonic() - started
        remaining = poll_period - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
