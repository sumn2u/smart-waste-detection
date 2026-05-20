import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

def import_onnxruntime_quietly():
    # ONNX Runtime can write harmless GPU-probing warnings directly to fd 2
    # during import, before Python-level logging controls exist.
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        import onnxruntime as imported_ort
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
        os.close(devnull)
    return imported_ort


@contextlib.contextmanager
def suppress_native_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
        os.close(devnull)


os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
with suppress_native_stderr():
    ort = import_onnxruntime_quietly()


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rtdetr.onnx"

ID2LABEL = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic"}
CONF_THRESHOLD = 0.5
INPUT_SIZE = (640, 640)

CAMERA_SIZES = (
    (480, 640),
    (240, 320),
)


def load_onnx_model(path):
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    print(f"Loading ONNX model from {path}...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    session = ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded successfully.")
    return session


class Picamera2Capture:
    def __init__(self, camera_index, width, height, color_format):
        from picamera2 import Picamera2

        self.color_format = color_format
        self.camera = Picamera2(camera_num=camera_index, verbose_console=False)
        config = self.camera.create_preview_configuration(
            main={"size": (width, height), "format": color_format},
            controls={"FrameRate": 15.0},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)

    def read(self):
        frame = self.camera.capture_array()
        if frame is None:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif self.color_format == "RGB888":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self):
        with contextlib.suppress(Exception):
            self.camera.stop()
        with contextlib.suppress(Exception):
            self.camera.close()


def open_picamera2(camera_index, color_format):
    if not sys.platform.startswith("linux"):
        return None, "Picamera2 is only used on Linux/Raspberry Pi."

    try:
        from picamera2 import Picamera2

        with suppress_native_stderr():
            cameras = Picamera2.global_camera_info()
    except Exception as exc:
        return None, f"Picamera2 import/check failed: {exc}"

    if not cameras:
        return None, "Picamera2/libcamera found no cameras"
    if camera_index >= len(cameras):
        return None, f"Picamera2 camera index {camera_index} is out of range; found {len(cameras)} camera(s)"

    errors = []
    for width, height in CAMERA_SIZES:
        try:
            with suppress_native_stderr():
                cap = Picamera2Capture(camera_index, width, height, color_format)
                ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"Camera opened with Picamera2/libcamera at {width}x{height} using {color_format}.")
                return cap, None
            cap.release()
            errors.append(f"Picamera2 {width}x{height}: opened but no frames")
        except Exception as exc:
            errors.append(f"Picamera2 {width}x{height}: {exc}")

    return None, "; ".join(errors)


def configure_camera(cap, width, height, fourcc):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 15)


def grab_test_frame(cap):
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    return None


def open_opencv_camera(camera_index):
    errors = []
    formats = ("MJPG", "YUYV")
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    backend_name = "V4L2" if sys.platform.startswith("linux") else "OpenCV"

    for width, height in CAMERA_SIZES:
        for fourcc in formats:
            with suppress_native_stderr():
                cap = cv2.VideoCapture(camera_index, backend)
                opened = cap.isOpened()

            if not opened:
                cap.release()
                errors.append(f"{backend_name} {width}x{height} {fourcc}: could not open")
                continue

            with suppress_native_stderr():
                configure_camera(cap, width, height, fourcc)
                frame = grab_test_frame(cap)

            if frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"Camera opened with {backend_name}, requested {width}x{height} {fourcc}, "
                    f"actual {actual_w}x{actual_h}."
                )
                return cap, None

            cap.release()
            time.sleep(0.2)
            errors.append(f"{backend_name} {width}x{height} {fourcc}: opened but no frames")

    return None, "; ".join(errors)


def open_camera(camera_index, backend, picamera_format):
    errors = []

    if backend in ("auto", "picamera2"):
        cap, error = open_picamera2(camera_index, picamera_format)
        if cap is not None:
            return cap
        errors.append(error or "Picamera2 failed")
        if backend == "picamera2":
            print_camera_error(errors, camera_index)
            return None

    if backend in ("auto", "opencv"):
        cap, error = open_opencv_camera(camera_index)
        if cap is not None:
            return cap
        errors.append(error or "OpenCV failed")

    print_camera_error(errors, camera_index)
    return None


def print_camera_error(errors, camera_index):
    print("Error: could not read frames from the camera.")
    print("Tried:")
    for item in errors:
        for part in str(item).split("; "):
            print(f"  - {part}")
    print("\nRaspberry Pi checks that often help:")
    print("  - Close any other program using the camera, including old debug sessions.")
    print("  - Test the Pi camera outside Python: libcamera-hello --timeout 2000")
    print("  - For a USB camera, test devices with: ls /dev/video*")
    print(
        "  - Force a backend if needed: "
        f"python conveyor_belt/camera_detection_fixed.py --backend picamera2 --camera {camera_index}"
    )


def predict_frame(session, frame):
    orig_h, orig_w = frame.shape[:2]

    img_resized = cv2.resize(frame, INPUT_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))[None, :, :, :])

    input_name = session.get_inputs()[0].name
    logits, boxes = session.run(None, {input_name: tensor})[:2]

    probs = 1 / (1 + np.exp(-logits[0]))
    labels = np.argmax(probs, axis=-1)
    scores = np.max(probs, axis=-1)

    keep = scores > CONF_THRESHOLD
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[0][keep]

    results = []
    for score, label, box in zip(scores, labels, boxes):
        cx, cy, nw, nh = box
        x1 = int((cx - nw / 2) * orig_w)
        y1 = int((cy - nh / 2) * orig_h)
        x2 = int((cx + nw / 2) * orig_w)
        y2 = int((cy + nh / 2) * orig_h)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        results.append((ID2LABEL.get(int(label), "unknown"), float(score), [x1, y1, x2, y2]))

    return results


def draw_detections(frame, detections, fps):
    for label, score, box in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2%}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RT-DETR waste detection from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 1.")
    parser.add_argument(
        "--backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="Camera backend. Auto tries Picamera2 first on Raspberry Pi, then OpenCV V4L2.",
    )
    parser.add_argument(
        "--picamera-format",
        choices=("RGB888", "BGR888"),
        default="RGB888",
        help="Picamera2 color format. Default RGB888 is converted to OpenCV BGR for correct display.",
    )
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow.")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = load_onnx_model(MODEL_PATH)
    cap = open_camera(args.camera, args.backend, args.picamera_format)
    if cap is None:
        return 1

    print("\n--- ONNX RUNTIME DEBUGGER ACTIVE ---")
    if args.no_window:
        print("Running without a video window. Press Ctrl+C to exit.\n")
    else:
        print("Press 'q' in the VIDEO WINDOW to exit.\n")

    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                return 1

            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            detections = predict_frame(detector, frame)
            if detections:
                debug_list = [f"{label} ({conf:.1%})" for label, conf, _ in detections]
                print(f"[FPS: {fps:.1f}] Objects: {', '.join(debug_list)}")

            draw_detections(frame, detections, fps)

            if not args.no_window:
                cv2.imshow("Smart Bin Camera Simulation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nShutting down...")
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
