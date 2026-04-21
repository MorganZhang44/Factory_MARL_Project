"""
train_mappo.py
Training entry point for MAPPO pursuit agent.

Usage:
    python scripts/train_mappo.py
    python scripts/train_mappo.py --config configs/mappo_config.yaml --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.trainers.mappo_trainer import MAPPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train MAPPO pursuit agent")
    p.add_argument("--config",   default="configs/mappo_config.yaml", help="Path to config YAML")
    p.add_argument("--save-dir", default="results/checkpoints",        help="Checkpoint directory")
    p.add_argument("--device",   default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume",   default=None,   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = MAPPOTrainer(cfg, device=args.device)

    if args.resume:
        trainer.load(args.resume)

    trainer.train(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
