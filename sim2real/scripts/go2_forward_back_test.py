from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SIM2REAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SDK_ROOT = SIM2REAL_ROOT / "unitree_sdk2_python"
if LOCAL_SDK_ROOT.exists():
    sys.path.insert(0, str(LOCAL_SDK_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Go2 forward at 0.1 m/s for 1 second, backward for 1 second, then stop."
    )
    parser.add_argument("--iface", default="eno1", help="Network interface connected to the robot.")
    parser.add_argument("--speed", type=float, default=0.1, help="Linear speed in m/s.")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration for each leg in seconds.")
    parser.add_argument(
        "--skip-standup",
        action="store_true",
        help="Do not send StandUp before moving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Safety check: please ensure the robot has clear space in front and behind it.", flush=True)
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.iface)

    client = SportClient()
    client.SetTimeout(10.0)
    client.Init()

    if not args.skip_standup:
        print("Standing up...", flush=True)
        code = client.StandUp()
        print(f"StandUp ret={code}", flush=True)
        time.sleep(2.0)
        print("Entering balance stand...", flush=True)
        code = client.BalanceStand()
        print(f"BalanceStand ret={code}", flush=True)
        time.sleep(1.0)

    def drive_for(vx: float, duration: float, hz: float = 20.0) -> None:
        period = 1.0 / hz
        deadline = time.monotonic() + duration
        last_code = None
        while time.monotonic() < deadline:
            last_code = client.Move(float(vx), 0.0, 0.0)
            time.sleep(period)
        print(f"Move stream ret={last_code}", flush=True)

    print(f"Moving forward at {args.speed:.3f} m/s for {args.duration:.2f}s", flush=True)
    drive_for(float(args.speed), float(args.duration))

    print(f"Moving backward at {-args.speed:.3f} m/s for {args.duration:.2f}s", flush=True)
    drive_for(float(-args.speed), float(args.duration))

    print("Stopping...", flush=True)
    code = client.StopMove()
    print(f"StopMove ret={code}", flush=True)
    time.sleep(0.5)

    print("Damping to finish cleanly...", flush=True)
    code = client.Damp()
    print(f"Damp ret={code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
