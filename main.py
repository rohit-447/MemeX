import os
import cv2
import numpy as np
import torch
import argparse

from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance, loadImage  # assuming correct

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

def detect(net, image_path, canvas_size=1280, mag_ratio=2.0, text_threshold=0.05, link_threshold=0.05, low_text=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = loadImage(image_path)
    orig_h, orig_w = image.shape[0:2]

    # resize with mag_ratio
    img_resized, target_ratio, _ = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = 1 / target_ratio
    ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x_t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = net(x_t)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    print(f"DEBUG: score_text — min {score_text.min():.6f}, max {score_text.max():.6f}, mean {score_text.mean():.6f}")
    print(f"DEBUG: score_link — min {score_link.min():.6f}, max {score_link.max():.6f}, mean {score_link.mean():.6f}")

    # Use official box extraction
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

    # Filter polys None etc
    final_boxes = []
    for b in boxes:
        final_boxes.append(np.array(b).reshape(-1,2))

    return image, final_boxes

def draw_and_save(image, boxes, output_path):
    img_copy = image.copy()
    for box in boxes:
        box = box.astype(int).reshape(-1,1,2)
        cv2.polylines(img_copy, [box], True, (0,255,0), 2)
    cv2.imwrite(output_path, img_copy)
    print(f"[INFO] Saved detection image to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--model', default='weights/craft_mlt_25k.pth', help='Model path')
    parser.add_argument('--output', default='detected.jpg', help='Output image')
    args = parser.parse_args()

    device = torch.device('cpu')
    net = load_craft(args.model, device)

    # Try with more aggressive sensitivity
    image, boxes = detect(net, args.image, canvas_size=1280, mag_ratio=2.5, text_threshold=0.02, link_threshold=0.02, low_text=0.005)

    if len(boxes) == 0:
        print("[INFO] Still no boxes detected with more sensitive settings.")
    else:
        print(f"[INFO] Detected {len(boxes)} boxes.")
        draw_and_save(image, boxes, args.output)

if __name__ == '__main__':
    main()
