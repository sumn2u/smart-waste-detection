import torch
import numpy as np
from PIL import Image
from pathlib import Path

# 1. Setup Paths dynamically
# This gets the directory where THIS script is saved
BASE_DIR = Path(__file__).resolve().parent

# Define paths relative to the script location
MODEL_PATH = BASE_DIR / "rtdetr_traced.pt"
IMAGE_PATH = BASE_DIR.parent / "samples" / "plastic_2.jpg"

# 2. Configuration
ID2LABEL = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}
CONF_THRESHOLD = 0.5

def load_lite_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = torch.jit.load(str(path), map_location='cpu')
    model.eval()
    return model

def simple_predict(model, img_path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found at: {img_path}")

    # --- PREPROCESSING ---
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    
    img_resized = img.resize((640, 640))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # Standardization
    # mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    # img_np = (img_np - mean) / std
    
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)

    # Match model precision (solves the Double vs Float error)
    model_dtype = next(model.parameters()).dtype
    tensor = tensor.to(model_dtype)

    # --- INFERENCE ---
    with torch.no_grad():
        logits, boxes = model(tensor)

    # --- POST-PROCESSING ---
    probs = logits.sigmoid()
    scores, labels = torch.max(probs[0], dim=-1)
    
    keep = scores > CONF_THRESHOLD
    scores, labels, boxes = scores[keep], labels[keep], boxes[0][keep]

    results = []
    for i in range(len(scores)):
        # Convert cxcywh (normalized) to xyxy (pixel coordinates)
        cx, cy, nw, nh = boxes[i].tolist()
        x1 = (cx - nw / 2) * orig_w
        y1 = (cy - nh / 2) * orig_h
        x2 = (cx + nw / 2) * orig_w
        y2 = (cy + nh / 2) * orig_h
        
        results.append({
            "label": ID2LABEL.get(labels[i].item(), "unknown"),
            "score": scores[i].item(),
            "box": [round(x, 2) for x in [x1, y1, x2, y2]]
        })
    return results

if __name__ == "__main__":
    try:
        detector = load_lite_model(MODEL_PATH)
        detections = simple_predict(detector, IMAGE_PATH)
        
        print(f"\nScanning: {IMAGE_PATH.name}")
        if not detections:
            print("No objects found.")
        for d in detections:
            print(f"[{d['label']}] Confidence: {d['score']:.2%} | Box: {d['box']}")
            
    except Exception as e:
        print(f"Status: Error - {e}")