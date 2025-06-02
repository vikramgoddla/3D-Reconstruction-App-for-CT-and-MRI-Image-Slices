import argparse
import os
import cv2
import numpy as np
import nibabel as nib
from ultralytics import YOLO
import torch
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------
# 1. Simple 2D CNN Regressor for (dx, dy, scale)
# --------------------------------------------
class SliceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128), nn.ReLU(),
            nn.Linear(128, 3)  # dx, dy, scale
        )

    def forward(self, x):
        return self.net(x)

def apply_transform(slice_img, dx, dy, scale):
    h, w = slice_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=0, scale=scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(slice_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# Models
model = YOLO('grid_image.pt')
model_screen = YOLO('screen_recording_detect.pt')
model_phone = YOLO('phone_recording_detect.pt')

STANDARD_SIZE = (256, 256)
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_video(filename):
    return filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def process_image_grid(image_path, output_name):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Failed to read image"
    # Assume fixed grid layout, e.g., 6x6
    rows, cols = 6, 6
    h, w = img.shape[0] // rows, img.shape[1] // cols
    slices = [cv2.resize(img[i*h:(i+1)*h, j*w:(j+1)*w], STANDARD_SIZE)
              for i in range(rows) for j in range(cols)]
    volume = np.stack(slices, axis=0)
    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    out_path = os.path.join(OUTPUT_FOLDER, f'{output_name}_grid.nii.gz')
    nib.save(nifti, out_path)
    print(f"Saved to {out_path}")

def process_video(video_path, output_name, model, expected_slices=36, align=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // expected_slices)

    slices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            results = model.predict(frame, conf=0.25, imgsz=640)[0]
            if len(results.boxes.xyxy) == 1:
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
                slices.append(resized)
        frame_idx += 1

    cap.release()
    if not slices:
        print("[!] No valid detections.")
        return

    if align:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_align = SliceRegressor().to(device)
        model_align.load_state_dict(torch.load("3D_volume_aligner.pth", map_location=device))
        model_align.eval()

        aligned_slices = [slices[0]]
        for i in range(1, len(slices)):
            img = slices[i].astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            inp = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                dx, dy, scale = model_align(inp).squeeze().cpu().numpy()
            aligned = apply_transform(slices[i], dx, dy, scale)
            aligned_slices.append(aligned)
        slices = aligned_slices

    volume = np.stack(slices, axis=0)
    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    suffix = "_aligned" if align else ""
    out_path = os.path.join(OUTPUT_FOLDER, f'{output_name}{suffix}.nii.gz')
    nib.save(nifti, out_path)
    print(f"Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="3D Slice Reconstruction Demo")
    parser.add_argument("filename", help="Input image or video file path")
    parser.add_argument("--phone", action="store_true", help="Use phone-trained YOLO model")
    parser.add_argument("--align", action="store_true", help="Apply slice alignment model")
    args = parser.parse_args()

    filename = args.filename
    base_name = os.path.splitext(os.path.basename(filename))[0]

    if is_image(filename):
        process_image_grid(filename, base_name)
    elif is_video(filename):
        model = model_phone if args.phone else model_screen
        process_video(filename, base_name, model, align=args.align)
    else:
        print("Unsupported file type.")

if __name__ == "__main__":
    main()
