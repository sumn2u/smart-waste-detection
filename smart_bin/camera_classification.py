import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
# Make sure you export your PyTorch model to this ONNX path
MODEL_PATH = BASE_DIR / "smart_bin.onnx"

# Classes must match your training order
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# 1. Load the ONNX Session
def load_onnx_model(path):
    print(f"Loading ONNX Classification Model from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    # Initialize ONNX Runtime session
    # (Defaults to CPU; will use GPU automatically if available and configured)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session


# 2. NumPy-based Preprocessing Pipeline (Replaces torchvision.transforms)
def preprocess_frame(frame_rgb):
    # Resize to 640x640 (matching your PyTorch settings)
    resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and scale to [0, 1] (Equivalent to transforms.ToTensor())
    img_data = resized.astype(np.float32) / 255.0

    # Normalize (Equivalent to transforms.Normalize(mean=..., std=...))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    # Change HWC layout (Height, Width, Channels) to CHW layout (Channels, Height, Width)
    img_data = np.transpose(img_data, (2, 0, 1))

    # Add batch dimension: (1, C, H, W)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


# 3. Softmax implementation for NumPy
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def main():
    try:
        # Initialize ONNX Session
        session = load_onnx_model(MODEL_PATH)
        input_name = session.get_inputs()[0].name

        # Initialize Camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(
                "Error: Could not open webcam. Check Privacy & Security settings."
            )
            return

        print("\n--- CLASSIFICATION LIVE LOGS (ONNX) ---")
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
            input_tensor = preprocess_frame(frame_rgb)

            # Run ONNX Inference
            # outputs[0] contains the raw logits
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0]

            # Post-processing (Softmax and Max)
            probs = softmax(logits)
            idx = np.argmax(probs, axis=1)[0]
            score = probs[0][idx]

            label = CLASSES[idx]

            # --- TERMINAL LOGGING ---
            if frame_count % 15 == 0:
                print(
                    f"[FPS: {fps:4.1f}] Top Prediction: {label:10} | Confidence: {score:.2%}"
                )

            # --- VISUAL OVERLAY ---
            display_text = f"{label} ({score:.1%})"
            cv2.putText(
                frame,
                display_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            # Show the frame
            cv2.imshow("Smart Bin Classification Simulation", frame)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopping Camera Feed...")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)

    except Exception as e:
        print(f"\nRuntime Error: {e}")


if __name__ == "__main__":
    main()