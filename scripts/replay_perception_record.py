#!/usr/bin/env python3
"""Replay a recorded Core->Perception request against the local perception runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_ROOT = PROJECT_ROOT / "perception"
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from perception_service import PerceptionRuntime  # noqa: E402


def _load_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "payload" in data:
        payload = data["payload"]
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-dict payload")
        return payload
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _latest_request_in_dir(path: Path) -> Path:
    matches = sorted(path.glob("*.request.json"))
    if not matches:
        raise FileNotFoundError(f"No '*.request.json' files found in {path}")
    return matches[-1]


def _print_summary(result: dict[str, Any]) -> None:
    print(f"step={result.get('step')} timestamp={result.get('timestamp')}")
    dogs = result.get("dogs", {})
    for robot_id in sorted(dogs.keys()):
        dog = dogs[robot_id]
        print(
            f"{robot_id}: localized={dog.get('localized')} "
            f"xy_error_m={dog.get('xy_error_m')} "
            f"position_world={dog.get('position_world')} "
            f"velocity_world={dog.get('velocity_world')}"
        )
    intruder = result.get("intruder_estimate")
    print(f"intruder_estimate={intruder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded perception request offline.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(PROJECT_ROOT / "output" / "perception_records"),
        help="Path to a recorded *.request.json file, or a directory containing them.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device passed to PerceptionRuntime.")
    parser.add_argument(
        "--output",
        help="Optional JSON path to save the replay result. Defaults to '<input>.replay.result.json' for file input.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print the full replay result as formatted JSON after the summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    request_path = _latest_request_in_dir(input_path) if input_path.is_dir() else input_path
    payload = _load_record(request_path)

    runtime = PerceptionRuntime(device=args.device)
    result = runtime.estimate(payload)

    print(f"request={request_path}")
    _print_summary(result)

    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif request_path.suffixes[-2:] == [".request", ".json"]:
        output_path = request_path.with_name(request_path.name.replace(".request.json", ".replay.result.json"))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"saved_result={output_path}")

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
