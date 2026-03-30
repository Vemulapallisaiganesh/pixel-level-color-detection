import argparse
from pathlib import Path
# Run inference using trained weights on a selected COCO split.

import cv2
from ultralytics import YOLO


def list_images(folder: Path):
    """Collect supported image files from the given folder."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(files)


def parse_args():
    """Parse CLI arguments for COCO-only batch inference."""
    parser = argparse.ArgumentParser(
        description="Run inference only on COCO dataset images and save outputs"
    )
    parser.add_argument(
        "--weights",
        default="runs/segment/coco2017_seg_custom/weights/best.pt",
        help="Trained model weights",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/coco2017",
        help="Path containing COCO split folders",
    )
    parser.add_argument(
        "--split",
        default="val2017",
        choices=["train2017", "val2017", "test2017"],
        help="COCO split to process",
    )
    parser.add_argument(
        "--output-dir",
        default="output/coco_only",
        help="Where processed images are saved",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap. 0 means process all images",
    )
    return parser.parse_args()


def main():
    # Resolve runtime paths and validate required inputs.
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    split_dir = dataset_root / args.split
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    model = YOLO(str(weights_path))  # Load trained segmentation model.
    image_files = list_images(split_dir)

    if args.max_images > 0:
        image_files = image_files[: args.max_images]

    if not image_files:
        raise RuntimeError(f"No images found in: {split_dir}")

    print(f"Processing {len(image_files)} images from {split_dir}")

    for idx, image_path in enumerate(image_files, start=1):
        result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]
        rendered = result.plot()  # Draw detections and masks onto a rendered frame.

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), rendered)

        if idx % 25 == 0 or idx == len(image_files):
            print(f"Saved {idx}/{len(image_files)}")

    print(f"Done. COCO-only output folder: {output_dir}")


if __name__ == "__main__":
    main()
