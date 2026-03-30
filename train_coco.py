import argparse
from pathlib import Path

from ultralytics import YOLO
# COCO training utility: prepares dataset YAML and launches YOLOv8 segmentation training.

COCO_80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def create_coco_seg_yaml(dataset_root: Path) -> Path:
    """Generate a YOLO-compatible COCO segmentation YAML under dataset root."""
    train_dir = dataset_root / "train2017"
    val_dir = dataset_root / "val2017"
    ann_train = dataset_root / "annotations" / "instances_train2017.json"
    ann_val = dataset_root / "annotations" / "instances_val2017.json"

    # Ensure expected COCO image and annotation paths exist before training.
    missing = [
        str(p)
        for p in [train_dir, val_dir, ann_train, ann_val]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "COCO dataset is incomplete. Missing paths:\n" + "\n".join(missing)
        )

    yaml_path = dataset_root / "coco2017-seg.yaml"
    names_text = "\n".join([f"  {i}: {name}" for i, name in enumerate(COCO_80_NAMES)])

    yaml_content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: train2017\n"
        "val: val2017\n"
        "test: val2017\n"
        "\n"
        "names:\n"
        f"{names_text}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def parse_args():
    """Parse CLI options for segmentation training."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation on COCO 2017")
    parser.add_argument("--dataset-root", default="datasets/coco2017", help="Path containing train2017, val2017, annotations")
    parser.add_argument("--weights", default="yolov8m-seg.pt", help="Initial weights path")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device, e.g., cpu, 0")
    parser.add_argument("--project", default="runs/segment", help="Output project directory")
    parser.add_argument("--name", default="coco2017_seg_custom", help="Run name")
    return parser.parse_args()


def main():
    # Prepare training config, then start YOLO training job.
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()

    yaml_path = create_coco_seg_yaml(dataset_root)

    model = YOLO(args.weights)  # Initialize from base/pretrained weights.
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training complete. Best model should be at: {best_weights}")


if __name__ == "__main__":
    main()
