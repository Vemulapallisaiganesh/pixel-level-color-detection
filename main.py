from ultralytics import YOLO
import cv2
import numpy as np
import os

# ------------------ LOAD MODEL ------------------
model = YOLO("yolov8m-seg.pt")

# ------------------ MAIN FUNCTION ------------------
def process_image(
    image_path,
    output_path="output/result.jpg",
    conf_threshold=0.3,
    include_classes=None,
    exclude_classes=None,
    return_metrics=False,
):
    """Run segmentation on an image, paint masks, place labels, and optionally return metrics."""

    # Normalize class names once so comparisons are case-insensitive and fast.
    include_set = {c.strip().lower() for c in include_classes} if include_classes else None
    exclude_set = {c.strip().lower() for c in exclude_classes} if exclude_classes else set()

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image not found")

    output = image.copy()  # Keep original image untouched; draw all overlays on this copy.
    
    image_area = image.shape[0] * image.shape[1]
    metrics = {
        'total_objects_detected': 0,
        'objects_after_filter': 0,
        'avg_confidence': 0.0,
        'max_confidence': 0.0,
        'min_confidence': 1.0,
        'total_mask_coverage': 0.0,
        'model_accuracy': 0.0,
        'confidence_threshold': conf_threshold,
    }

    # Run model
    results = model(image, conf=conf_threshold)[0]  # Single-image inference result.

    if results.masks is not None:

        masks = results.masks.data.cpu().numpy()  # Segmentation masks in model output size.
        boxes = results.boxes  # Per-detection class IDs and confidence scores.
        
        # Track metrics
        all_confidences = []
        filtered_count = 0

        # Fixed colors for accuracy (BGR format)
        color_list = [
            (255, 0, 0),     # Blue
            (0, 255, 0),     # Green
            (0, 0, 255),     # Red
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Purple
            (255, 255, 0),   # Cyan
        ]

        # Color name mapping
        color_map = {
            (255, 0, 0): "Blue",
            (0, 255, 0): "Green",
            (0, 0, 255): "Red",
            (0, 255, 255): "Yellow",
            (255, 0, 255): "Purple",
            (255, 255, 0): "Cyan"
        }

        metrics['total_objects_detected'] = len(masks)

        for i, mask in enumerate(masks):

            cls_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            all_confidences.append(confidence)
            label = str(model.names[cls_id]).strip().lower()

            # Skip classes user does not want in the final rendering.
            if label in exclude_set:
                continue
            if include_set and label not in include_set:
                continue

            filtered_count += 1

            # Resize mask
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Align with original image.
            mask = (mask > 0.5).astype(np.uint8)  # Convert soft mask to binary mask.

            # Improve edges
            kernel = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Create clean mask
            clean_mask = np.zeros_like(mask)

            for cnt in contours:
                if cv2.contourArea(cnt) < 500:  # Ignore tiny blobs/noise.
                    continue
                cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=-1)

            if cv2.countNonZero(clean_mask) == 0:
                continue

            # Calculate mask coverage
            mask_pixels = cv2.countNonZero(clean_mask)
            mask_coverage = (mask_pixels / image_area) * 100
            metrics['total_mask_coverage'] += mask_coverage

            # ------------------ COLOR FILL ------------------
            color = color_list[i % len(color_list)]  # Cycle through fixed colors per object index.

            for c in range(3):
                output[:, :, c] = np.where(clean_mask == 1,
                                           color[c],
                                           output[:, :, c])

            # ------------------ LABEL ------------------
            label = model.names[cls_id]

            # ------------------ FIND CENTER ------------------
            ys, xs = np.where(clean_mask == 1)  # Pixel coordinates of this object region.

            if len(xs) == 0 or len(ys) == 0:
                continue

            center_x = int(np.mean(xs))  # Use mask centroid for centered label placement.
            center_y = int(np.mean(ys))

            # ------------------ TEXT ------------------
            text = f"{label}"

            # Medium professional font
            font_scale = max(0.6, image.shape[1] / 1000)  # Keep text readable on large/small images.
            thickness = max(2, int(font_scale * 2))

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            padding = 8

            # Background box
            cv2.rectangle(output,
                          (center_x - text_w//2 - padding,
                           center_y - text_h - padding),
                          (center_x + text_w//2 + padding,
                           center_y + padding),
                          (0, 0, 0),
                          -1)

            # Draw text
            cv2.putText(output,
                        text,
                        (center_x - text_w//2, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA)

        # Calculate final metrics
        metrics['objects_after_filter'] = filtered_count  # Objects that passed include/exclude filters.
        if all_confidences:
            metrics['avg_confidence'] = np.mean(all_confidences)
            metrics['max_confidence'] = np.max(all_confidences)
            metrics['min_confidence'] = np.min(all_confidences)
            metrics['model_accuracy'] = metrics['avg_confidence'] * 100  # Confidence proxy, not benchmark mAP.
        
        metrics['total_mask_coverage'] = round(metrics['total_mask_coverage'], 2)

    # ------------------ SAVE OUTPUT ------------------
    if output_path:
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, output)

    if return_metrics:
        return output, metrics
    return output