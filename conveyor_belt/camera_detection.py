import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime as ort

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rtdetr.onnx"  # Using the ONNX version
ID2LABEL = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}
CONF_THRESHOLD = 0.5

def load_lite_model(path):
    print(f"Loading ONNX model from {path}...")
    
    # Configure session options for Raspberry Pi CPU optimization
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # Matches Raspberry Pi 4/5 quad-core architecture
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    print("Model loaded successfully.")
    return session

def predict_frame(session, frame):
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Faster preprocessing with OpenCV (avoids slow PIL conversions)
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert from HWC (Height, Width, Channel) to CHW format
    img_np = img_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(img_np, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension: [1, 3, 640, 640]

    # 2. Run ONNX Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    
    # RT-DETR typically outputs [logits, boxes]
    logits, boxes = outputs[0], outputs[1]

    # 3. Fast Post-processing using NumPy vectors
    # Sigmoid function applied to logits: 1 / (1 + e^-x)
    probs = 1 / (1 + np.exp(-logits[0]))
    
    labels = np.argmax(probs, axis=-1)
    scores = np.max(probs, axis=-1)
    
    # Filter by confidence threshold
    keep = scores > CONF_THRESHOLD
    scores, labels, boxes = scores[keep], labels[keep], boxes[0][keep]

    results = []
    for i in range(len(scores)):
        cx, cy, nw, nh = boxes[i]
        
        # Convert Normalized CXCYWH format to Pixel X1Y1X2Y2 coordinates
        x1, y1 = (cx - nw/2) * orig_w, (cy - nh/2) * orig_h
        x2, y2 = (cx + nw/2) * orig_w, (cy + nh/2) * orig_h
        
        results.append((
            ID2LABEL.get(int(labels[i]), 'unknown'), 
            float(scores[i]), 
            [int(x1), int(y1), int(x2), int(y2)]
        ))
    return results

def main():
    detector = load_lite_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera access denied or not found.")
        return

    # 4. Limit camera resolution directly at the source (Massive CPU saver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n--- ONNX RUNTIME DEBUGGER ACTIVE ---")
    print("Press 'q' in the VIDEO WINDOW to exit.\n")
    
    prev_time = time.time()

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
    # Desktop environment specific cleanup
    for i in range(5): cv2.waitKey(1)

if __name__ == "__main__":
    main()