import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import time
import numpy as np

# --- SETUP ---
# BASE_DIR points to the folder where this script is located
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "smart_bin.pth"

# Classes must match your training order (6 classes based on your model file)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 1. Load the Model
def load_lite_model(path, num_classes=6):
    print(f"Loading Classification Model from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
        
    # Reconstruct the MobileNetV3 architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    # Use TorchScript (JIT) for optimized inference performance
    return torch.jit.script(model)

# 2. Define the Preprocessing Pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(), # Necessary for OpenCV BGR to Torchvision conversion
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    try:
        # Initialize Model
        model = load_lite_model(MODEL_PATH, num_classes=len(CLASSES))
        model_dtype = next(model.parameters()).dtype
        
        # Initialize Camera (0 is built-in webcam)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Check Privacy & Security settings.")
            return

        print("\n--- CLASSIFICATION LIVE LOGS ---")
        print("Click the video window and press 'q' to exit.\n")
        
        prev_time = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_time = curr_time

            # Preprocess the frame (OpenCV BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame_rgb).unsqueeze(0).to(model_dtype)

            # Run Model Inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, 1)
            
            label = CLASSES[idx.item()]
            score = conf.item()

            # --- TERMINAL LOGGING (ROWS) ---
            # Print a new row every 15 frames so the logs are readable
            if frame_count % 15 == 0:
                print(f"[FPS: {fps:4.1f}] Top Prediction: {label:10} | Confidence: {score:.2%}")

            # --- VISUAL OVERLAY ---
            # Draw the label and confidence on the video frame
            display_text = f"{label} ({score:.1%})"
            cv2.putText(frame, display_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 255, 0), 3)
            
            # Draw FPS counter
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2)

            # Show the frame
            cv2.imshow('Smart Bin Classification Simulation', frame)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping Camera Feed...")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        # Ensure macOS window closes
        for i in range(5):
            cv2.waitKey(1)
            
    except Exception as e:
        print(f"\nRuntime Error: {e}")

if __name__ == "__main__":
    main()