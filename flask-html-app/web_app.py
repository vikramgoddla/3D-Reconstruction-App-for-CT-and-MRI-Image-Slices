from flask import Flask, render_template, request, redirect, url_for, send_file, session, Response
import numpy as np
import cv2
import os
import json
import nibabel as nib
from ultralytics import YOLO
import threading
import pytesseract

app = Flask(__name__)
app.secret_key = 'ocr_4_life'
# Paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO models
model = YOLO('best.pt')
model_screen = YOLO('roi_aug_detector_final.pt')
model_phone = YOLO('ct_fullframe_detector_final.pt')

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    image = cv2.imread(filepath)
    height, width = image.shape[:2]
    return render_template('transform_drag.html', filename=file.filename, width=width, height=height)

@app.route('/transform', methods=['POST'])
def transform_image():
    filename = request.form['filename']
    coords = request.form.getlist('coords[]')
    coords = np.array(json.loads(coords[0]))

    if len(coords) != 4:
        return "Error: Exactly four coordinates are required."

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = cv2.imread(image_path)

    # Step 1: Perspective transform
    warped = four_point_transform(image, coords)

    # Step 2: YOLOv8 inference
    # Step 2: YOLOv8 inference
    results = model.predict(warped, conf=0.25, imgsz=640)[0]

    # Clone warped image for drawing
    debug_img = warped.copy()

    # Step 3: Crop detected slices and draw boxes
    slices = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = warped[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        slices.append((gray, x1, y1))

        # Draw bounding box
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Step 4: Save debug image
    debug_path = os.path.join(OUTPUT_FOLDER, filename + '_debug.jpg')
    cv2.imwrite(debug_path, debug_img)

    # Step 5: If no slices detected, show debug image instead of error
    if not slices:
        return send_file(debug_path, as_attachment=True, download_name="no_slices_detected.jpg")

    # Step 5: Sort and stack
    sorted_slices = sorted(slices, key=lambda tup: tup[2] * 10000 + tup[1])
    STANDARD_SIZE = (256, 256)  # or whatever you prefer

    volume = np.stack(
        [cv2.resize(s[0], STANDARD_SIZE, interpolation=cv2.INTER_AREA) for s in sorted_slices],
        axis=0
    )

    # Step 6: Save as NIfTI
    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    out_path = os.path.join(OUTPUT_FOLDER, filename + '_volume.nii.gz')
    nib.save(nifti, out_path)

    return send_file(out_path, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/output/<filename>')
def serve_output_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path) or os.path.getsize(path) < 50000:
        return '', 404
    return send_file(path)

def generate_live_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame, conf=0.25, imgsz=640)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/stream_phone_video/<filename>')
def stream_phone_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_live_video(video_path, model_phone), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_screen_video/<filename>')
def stream_screen_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_live_video(video_path, model_screen), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video_yolo_every_nth(video_path, model, filename, suffix='', expected_slices = 36):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // expected_slices)

    slices = []
    frame_idx = 0
    curr_idx = 0
    STANDARD_SIZE = (256, 256)

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
        print(f"[DEBUG] No valid slices — skipping NIfTI creation for {filename}{suffix}")
        return

    volume = np.stack(slices, axis=0)
    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    out_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}{suffix}_volume.nii.gz")
    nib.save(nifti, out_path)
    print(f"[DEBUG] Saved NIfTI to {out_path}")

def run_nifti_processing_async(video_path, model, filename, suffix='', expected_slices=50):
    threading.Thread(target=process_video_yolo_every_nth, args=(video_path, model, filename, suffix, expected_slices)).start()

@app.route('/upload_video_screen', methods=['POST'])
def upload_video_screen():
    file = request.files.get('video')
    expected_slices = int(request.form.get('slice_count', 36))
    if not file or file.filename == '':
        return redirect(request.url)
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    session['ocr_video_filename'] = filename
    return redirect(url_for('select_ocr_roi'))

@app.route('/select_ocr_roi')
def select_ocr_roi():
    filename = session.get('ocr_video_filename')
    if not filename:
        return "No filename in session", 400

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"[DEBUG] Trying to load: {video_path}")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Failed to load video frame", 400

    frame_path = os.path.join(UPLOAD_FOLDER, "first_frame.jpg")
    cv2.imwrite(frame_path, frame)

    return render_template("ocr_selector.html", frame_path=url_for('uploaded_file', filename="first_frame.jpg"), filename=filename)

@app.route('/submit_ocr_roi', methods=['POST'])
def submit_ocr_roi():
    from alignment_model import SliceRegressor, apply_transform
    import torch

    x = int(request.form['x'])
    y = int(request.form['y'])
    w = int(request.form['w'])
    h = int(request.form['h'])
    filename = request.form['video_filename']
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    cap = cv2.VideoCapture(video_path)
    last_text = None
    frames = []
    STANDARD_SIZE = (256, 256)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_crop = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(roi_gray, config='--psm 7').strip()

        if text and text != last_text:
            results = model_screen.predict(frame, conf=0.25, imgsz=640)[0]
            if len(results.boxes.xyxy) == 1:
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
                frames.append(resized)
                last_text = text

    cap.release()

    if not frames:
        return "No valid OCR/YOLO slices detected."

    # Apply alignment model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceRegressor().to(device)
    model.load_state_dict(torch.load("slice_transform_regressor.pth", map_location=device))
    model.eval()

    aligned_slices = [frames[0]]
    for i in range(1, len(frames)):
        img = frames[i].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        inp = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            dx, dy, scale = model(inp).squeeze().cpu().numpy()
        aligned = apply_transform(frames[i], dx, dy, scale)
        aligned_slices.append(aligned)

    volume = np.stack(aligned_slices, axis=0)
    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    out_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_ocr_aligned.nii.gz")
    nib.save(nifti, out_path)

    return redirect(url_for('serve_output_file', filename=os.path.basename(out_path)))

@app.route('/upload_video_phone', methods=['POST'])
def upload_video_phone():
    file = request.files.get('video')
    expected_slices = int(request.form.get('slice_count',36))
    if not file or file.filename == '':
        return redirect(request.url)
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    run_nifti_processing_async(filepath, model_phone, filename, suffix='_phone', expected_slices=expected_slices)
    nifti_filename = f"{os.path.splitext(filename)[0]}_phone_volume.nii.gz"
    return render_template('stream_viewer.html', stream_url=url_for('stream_phone_video', filename=filename), nifti_filename=nifti_filename)

if __name__ == '__main__':
    app.run(debug=True)
