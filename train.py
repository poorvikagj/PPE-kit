"""Train and validate the PPE detector with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

TORCHVISION_BY_TORCH_PREFIX = {
    "2.7": "0.22",
}


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/data.yaml")
    parser.add_argument("--base-model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu, 0, cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="models/training_runs")
    parser.add_argument("--name", default="ppe_yolo_finetune")
    parser.add_argument("--weights-out", default="models/weights/ppe_best.pt")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def validate_torchvision_compatibility() -> None:
    """Fail early when torch and torchvision are not a compatible pair."""
    import torch
    import torchvision

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    torchvision_version = torchvision.__version__.split("+", maxsplit=1)[0]
    torch_prefix = ".".join(torch_version.split(".")[:2])
    expected_prefix = TORCHVISION_BY_TORCH_PREFIX.get(torch_prefix)

    if expected_prefix and not torchvision_version.startswith(expected_prefix):
        raise RuntimeError(
            "Incompatible torch/torchvision install: "
            f"torch=={torch.__version__}, torchvision=={torchvision.__version__}. "
            f"For torch {torch_version}, install torchvision {expected_prefix}.x, for example: "
            "python -m pip install --force-reinstall torchvision==0.22.1"
        )


def configure_ultralytics_home() -> None:
    """Keep Ultralytics settings/cache inside the project workspace."""
    config_dir = Path(".ultralytics")
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))


def main() -> None:
    """Run a reproducible fine-tuning job and copy best weights."""
    args = parse_args()
    configure_ultralytics_home()
    validate_torchvision_compatibility()

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=1e-3,
        patience=15,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        project=args.project,
        name=args.name,
        seed=42,
        device=args.device,
        workers=args.workers,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    output_path = Path(args.weights_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    print(f"Copied best weights to {output_path}")

    trained_model = YOLO(str(output_path))
    metrics = trained_model.val(data=args.data, imgsz=args.imgsz)
    print(metrics)

    if args.export_onnx:
        trained_model.export(format="onnx")


if __name__ == "__main__":
    main()
