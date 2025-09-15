import sys
import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np

from craft import CRAFT
import imgproc
import craft_utils
import time

def copyStateDict(state_dict):
    """Remove 'module.' prefix if present in the state dict keys"""
    if list(state_dict.keys())[0].startswith("module."):
        print("Removing 'module.' prefix from state dict keys")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        return new_state_dict
    else:
        return state_dict

def save_heatmap(score_text, filename):
    heatmap = score_text
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(filename, heatmap)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--weights', default='weights/craft_mlt_25k.pth', help='Pretrained weights')
    parser.add_argument('--canvas_size', type=int, default=1280, help='Image size for inference')
    parser.add_argument('--mag_ratio', type=float, default=1.5, help='Image magnification ratio')
    parser.add_argument('--text_threshold', type=float, default=0.7, help='Text confidence threshold')
    parser.add_argument('--link_threshold', type=float, default=0.4, help='Link confidence threshold')
    parser.add_argument('--low_text', type=float, default=0.4, help='Low text threshold')
    args = parser.parse_args()

    # Device
    device = torch.device('cpu')
    print("[INFO] Using device:", device)

    # Load image
    image = imgproc.loadImage(args.image)
    print(f"[INFO] Loaded image shape: {image.shape}")

    # Load model
    net = CRAFT()  # initialize
    net.load_state_dict(copyStateDict(torch.load(args.weights, map_location=device)))
    net.to(device)
    net.eval()

    # Preprocess image
    resized_image, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Normalize and prepare tensor
    x = imgproc.normalizeMeanVariance(resized_image)
    x = torch.from_numpy(x).permute(2, 0, 1)  # HWC to CHW
    x = x.unsqueeze(0).to(device)  # add batch dim

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().numpy()
    score_link = y[0,:,:,1].cpu().numpy()

    # Save heatmap for visualization
    save_heatmap(score_text, "heatmap.jpg")
    print("[INFO] Heatmap saved as heatmap.jpg")

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, args.low_text)

    # Adjust boxes according to original image size
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    # Filter out small boxes to reduce noise
    filtered_boxes = []
    min_box_area = 300  # Tune this threshold to your liking
    for box in boxes:
        box = box.astype(np.int32)
        area = cv2.contourArea(box)
        if area > min_box_area:
            filtered_boxes.append(box)

    # Draw boxes on original image
    for box in filtered_boxes:
        cv2.polylines(image, [box.reshape((-1,1,2))], True, (0,255,0), 2)

    # Save result image
    result_path = "result.jpg"
    cv2.imwrite(result_path, image)
    print(f"[INFO] Result saved to {result_path}")

if __name__ == '__main__':
    main()
