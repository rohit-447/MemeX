import os
import cv2
import numpy as np
import torch
import argparse

from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance, loadImage

# Try optional pytesseract (best results). If not present, we'll fallback.
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


# ------------------------------
# Utils
# ------------------------------
def remove_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
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
    filtered = []
    for b in boxes:
        rect = b.reshape(-1, 2).astype(int)
        x, y, w, h = cv2.boundingRect(rect)
        if w * h >= min_area:
            filtered.append(b)
    return filtered


def filter_boxes_by_aspect_ratio(boxes, min_ratio=2.0):
    filtered = []
    for b in boxes:
        rect = b.reshape(-1, 2).astype(int)
        x, y, w, h = cv2.boundingRect(rect)
        ratio = w / float(h + 1e-6)
        if ratio > min_ratio:
            filtered.append(b)
    return filtered


def merge_close_boxes(boxes, overlap_thresh=0.2):
    if len(boxes) == 0:
        return []

    rects = [cv2.boundingRect(b.reshape(-1, 2).astype(int)) for b in boxes]
    rects_np = np.array(rects)

    # Prepare dummy scores for NMS
    scores = [1.0] * len(rects_np)
    indices = cv2.dnn.NMSBoxes(rects_np.tolist(), scores, score_threshold=0.0, nms_threshold=overlap_thresh)

    if indices is None or len(indices) == 0:
        return []

    merged = []
    for i in indices:
        # indices may be [[0], [2], ...] or array([0,2,...])
        idx = i if isinstance(i, (int, np.integer)) else int(i[0])
        x, y, w, h = rects_np[idx]
        merged.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    return merged


# ------------------------------
# Font-size estimation helpers
# ------------------------------
def estimate_font_size_tesseract(crop_bgr):
    """
    Use pytesseract to get word-level bounding boxes and return median height.
    crop_bgr: BGR image (numpy)
    """
    # Convert to RGB for pytesseract
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Optional: If Tesseract is not in PATH on Windows the user must set:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    data = pytesseract.image_to_data(rgb, output_type=Output.DICT, config="--psm 6")
    heights = []
    n = len(data["level"])
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if txt == "":
            continue
        h = int(data["height"][i])
        w = int(data["width"][i])
        # filter nonsense tiny boxes
        if h >= 4 and w >= 4:
            heights.append(h)
    if len(heights) > 0:
        return int(np.median(heights))
    return None


def estimate_font_size_contours(crop_bgr):
    """
    Fallback method without OCR:
      - adaptive threshold + morphological ops
      - find contours, take median contour bbox height
      - also try horizontal projection to detect line count and estimate
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h_crop, w_crop = gray.shape[:2]
    if h_crop <= 2 or w_crop <= 2:
        return max(1, int(h_crop / 2))

    # Adaptive threshold (inverted so text will be white on black)
    try:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 9)
    except Exception:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Small closing to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Horizontal projection to estimate number of text lines
    hor_proj = np.sum(bw // 255, axis=1)  # number of white pixels per row
    thr = max(1, int(0.15 * np.max(hor_proj)))  # threshold
    rows = np.where(hor_proj > thr)[0]
    line_count = 0
    if rows.size > 0:
        segments = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
        line_count = len(segments)

    # Contour approach: find contours and measure heights
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = []
    min_contour_area = max(10, (h_crop * w_crop) * 0.0003)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_contour_area and h >= 3:
            heights.append(h)

    # Prefer median contour height if we have it
    if len(heights) > 0:
        med_h = int(np.median(heights))
        # If we also detected line_count, sanity-check: per-line estimate shouldn't be much smaller than med_h
        if line_count >= 1:
            per_line_est = int(h_crop / max(1, line_count) / 1.15)
            # choose a robust estimate between contour median and per-line estimate
            # (favor the value that makes sense relative to crop)
            if 0 < med_h < per_line_est * 3:
                return med_h
            return per_line_est
        return med_h

    # If no contours found, use the line_count fallback
    if line_count >= 1:
        return max(1, int(h_crop / float(line_count) / 1.15))

    # last resort: use a fraction of the crop height
    return max(1, int(h_crop / 3.0))


def estimate_font_size(crop_bgr):
    """
    Try Tesseract first (if available) otherwise fallback to contour method.
    Returns integer px estimate.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0

    # Try tesseract for accurate per-word heights
    if TESSERACT_AVAILABLE:
        try:
            est = estimate_font_size_tesseract(crop_bgr)
            if est is not None:
                return est
        except Exception:
            # any tesseract error -> fallback
            pass

    # Fallback contour-based estimate
    return estimate_font_size_contours(crop_bgr)


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

    # Load original color image (for drawing)
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_h, orig_w = orig_image.shape[0:2]

    # Normalized version for detection (CRAFT expects the special loader)
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

    # Official extraction
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    final_boxes = [np.array(b).reshape(-1, 2) for b in boxes]

    # Apply filters
    filtered = filter_boxes_by_area(final_boxes, min_area=5000)
    filtered = filter_boxes_by_aspect_ratio(filtered, min_ratio=2.0)
    filtered = merge_close_boxes(filtered, overlap_thresh=0.2)

    # Compute font-size estimates per region
    sizes = []
    for i, poly in enumerate(filtered):
        poly_pts = poly.reshape(-1, 2).astype(int)
        x, y, w, h = cv2.boundingRect(poly_pts)
        # clamp crop coordinates to image boundaries
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(orig_w, x + w), min(orig_h, y + h)
        crop = orig_image[y0:y1, x0:x1].copy()
        if crop.size == 0:
            font_est = 0
        else:
            font_est = estimate_font_size(crop)

        sizes.append({
            "box_id": i,
            "x": int(x0),
            "y": int(y0),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
            "font_size_est": int(font_est),
        })
        print(f"[INFO] Box {i}: bbox(w={x1-x0}, h={y1-y0}), font≈{font_est}px")

    return orig_image, filtered, sizes


# ------------------------------
# Drawing
# ------------------------------
def draw_and_save(image, boxes, sizes, output_path):
    img_copy = image.copy()
    for box, size_info in zip(boxes, sizes):
        poly_pts = box.reshape(-1, 2).astype(int)
        cv2.polylines(img_copy, [poly_pts], True, (0, 255, 0), 2)

        x, y, w, h = size_info["x"], size_info["y"], size_info["width"], size_info["height"]
        label = f"{size_info['font_size_est']}px"
        # place label above box if there's room, otherwise inside
        label_x, label_y = x, max(12, y - 8)
        cv2.putText(
            img_copy,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img_copy)
    print(f"[INFO] Saved annotated image to: {output_path}")


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

    image, boxes, sizes = detect(
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
        draw_and_save(image, boxes, sizes, args.output)


if __name__ == "__main__":
    main()
