import os
import cv2
import numpy as np
import torch
import argparse

from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance, loadImage  # assuming correct


# ------------------------------
# Utils
# ------------------------------
def remove_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state


def load_craft(model_path, device):
    net = CRAFT()
    state = torch.load(model_path, map_location=device)
    state = remove_module_prefix(state)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


# ------------------------------
# Filtering Functions
# ------------------------------
def filter_boxes_by_area(boxes, min_area=5000):
    """Remove very small detections (noise, usernames, icons)."""
    filtered = []
    for b in boxes:
        x, y, w, h = cv2.boundingRect(b.astype(int))
        if w * h >= min_area:
            filtered.append(b)
    return filtered


def filter_boxes_by_aspect_ratio(boxes, min_ratio=2.0):
    """Keep wide rectangles (likely main text)."""
    filtered = []
    for b in boxes:
        x, y, w, h = cv2.boundingRect(b.astype(int))
        ratio = w / float(h)
        if ratio > min_ratio:  # tweet lines are wide
            filtered.append(b)
    return filtered


def merge_close_boxes(boxes, overlap_thresh=0.2):
    """Merge overlapping/close boxes into larger text regions."""
    if len(boxes) == 0:
        return []

    rects = [cv2.boundingRect(b.astype(int)) for b in boxes]
    rects_np = np.array(rects)

    indices = cv2.dnn.NMSBoxes(
        rects_np.tolist(), [1] * len(rects_np), score_threshold=0.0, nms_threshold=overlap_thresh
    )

    if len(indices) == 0:
        return []

    merged = []
    for i in indices:
        # Handle both scalar (int) and nested list (e.g. [0])
        idx = i if isinstance(i, (int, np.integer)) else i[0]
        x, y, w, h = rects_np[idx]
        merged.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    return merged



# ------------------------------
# Detection
# ------------------------------
def detect(
    net,
    image_path,
    canvas_size=1280,
    mag_ratio=2.0,
    text_threshold=0.05,
    link_threshold=0.05,
    low_text=0.01,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load color image for drawing
    orig_image = cv2.imread(image_path)   # <-- keeps color (BGR)
    orig_h, orig_w = orig_image.shape[0:2]

    # For detection, still use loadImage (normalized RGB float)
    image = loadImage(image_path)
    
    # resize with mag_ratio
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = 1 / target_ratio
    ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = net(x_t)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Use official box extraction
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    final_boxes = [np.array(b).reshape(-1, 2) for b in boxes]

    # ------------------------------
    # Apply Filters
    # ------------------------------
    filtered = filter_boxes_by_area(final_boxes, min_area=5000)
    filtered = filter_boxes_by_aspect_ratio(filtered, min_ratio=2.0)
    filtered = merge_close_boxes(filtered, overlap_thresh=0.2)

    return orig_image, filtered   # return the original color image



# ------------------------------
# Drawing
# ------------------------------
def draw_and_save(image, boxes, output_path):
    img_copy = image.copy()
    for box in boxes:
        box = box.astype(int).reshape(-1, 1, 2)
        cv2.polylines(img_copy, [box], True, (0, 255, 0), 2)
    cv2.imwrite(output_path, img_copy)
    print(f"[INFO] Saved detection image to {output_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--model", default="weights/craft_mlt_25k.pth", help="Model path")
    parser.add_argument("--output", default="detected.jpg", help="Output image")
    args = parser.parse_args()

    device = torch.device("cpu")
    net = load_craft(args.model, device)

    # More aggressive sensitivity for text
    image, boxes = detect(
        net,
        args.image,
        canvas_size=1280,
        mag_ratio=2.5,
        text_threshold=0.02,
        link_threshold=0.02,
        low_text=0.005,
    )

    if len(boxes) == 0:
        print("[INFO] No main text boxes detected after filtering.")
    else:
        print(f"[INFO] Detected {len(boxes)} main text boxes.")
        draw_and_save(image, boxes, args.output)


if __name__ == "__main__":
    main()
