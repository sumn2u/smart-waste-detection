import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import time
import numpy as np

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "smart_bin.pth"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 1. Load optimized model
def load_lite_model(path, num_classes=6):
    print(f"Loading Classification Model from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
        
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    # Scripting for faster CPU execution
    return torch.jit.script(model)

# 2. Preprocessing Transform
preprocess = transforms.Compose([
    transforms.ToPILImage(), # Convert OpenCV BGR array to PIL
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    try:
        model = load_lite_model(MODEL_PATH, num_classes=len(CLASSES))
        model_dtype = next(model.parameters()).dtype
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera access denied.")
            return

        print("\n--- CLASSIFICATION DEBUGGER ACTIVE ---")
        print("Press 'q' in the video window to exit.\n")
        
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Preprocess frame
            # OpenCV uses BGR; convert to RGB for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame_rgb).unsqueeze(0).to(model_dtype)

            # Inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, 1)
            
            label = CLASSES[idx.item()]
            score = conf.item()

            # --- TERMINAL DEBUG ---
            # Using \r (carriage return) keeps the debugger on one line for a cleaner look
            print(f"[FPS: {fps:4.1f}] Top Prediction: {label:10} | Confidence: {score:.2%}", end="\r")

            # --- VISUAL FEEDBACK ---
            # Display result on the camera frame
            display_text = f"{label} ({score:.1%})"
            cv2.putText(frame, display_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 255, 0), 3)
            
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2)

            cv2.imshow('MobileNet Smart Bin Classifier', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nShutting down...")
                break

        cap.release()
        cv2.destroyAllWindows()
        # Cleanup for macOS
        for i in range(5): cv2.waitKey(1)
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()