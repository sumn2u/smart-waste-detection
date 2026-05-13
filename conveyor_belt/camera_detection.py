import torch
import numpy as np
import cv2
import time  # Added for FPS calculation
from PIL import Image
from pathlib import Path

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rtdetr_traced.pt"
ID2LABEL = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}
CONF_THRESHOLD = 0.5

def load_lite_model(path):
    print(f"Loading model from {path}...")
    model = torch.jit.load(str(path), map_location='cpu')
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_frame(model, frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    orig_w, orig_h = pil_img.size
    
    img_resized = pil_img.resize((640, 640))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    model_dtype = next(model.parameters()).dtype
    tensor = tensor.to(model_dtype)

    with torch.no_grad():
        logits, boxes = model(tensor)

    probs = logits.sigmoid()
    scores, labels = torch.max(probs[0], dim=-1)
    
    keep = scores > CONF_THRESHOLD
    scores, labels, boxes = scores[keep], labels[keep], boxes[0][keep]

    results = []
    for i in range(len(scores)):
        cx, cy, nw, nh = boxes[i].tolist()
        x1, y1 = (cx - nw/2) * orig_w, (cy - nh/2) * orig_h
        x2, y2 = (cx + nw/2) * orig_w, (cy + nh/2) * orig_h
        results.append((ID2LABEL.get(labels[i].item()), scores[i].item(), [int(x1), int(y1), int(x2), int(y2)]))
    return results

def main():
    detector = load_lite_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera access denied or not found.")
        return

    print("\n--- DEBUGGER ACTIVE ---")
    print("Press 'q' in the VIDEO WINDOW to exit.\n")
    
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Run model
        detections = predict_frame(detector, frame)

        # --- TERMINAL DEBUGGER OUTPUT ---
        if detections:
            print(f"[FPS: {fps:.1f}] Objects: ", end="")
            debug_list = [f"{label} ({conf:.1%})" for label, conf, box in detections]
            print(", ".join(debug_list))
        else:
            # Optional: Print something even when empty to show it's still running
            # print(f"[FPS: {fps:.1f}] No detections", end="\r") 
            pass

        # Draw on frame
        for label, score, box in detections:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2%}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show FPS on the screen too
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Smart Bin Camera Simulation', frame)

        # Exit logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nShutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()
    # MacBook specific cleanup
    for i in range(5): cv2.waitKey(1)

if __name__ == "__main__":
    main()