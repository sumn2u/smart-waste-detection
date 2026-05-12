import argparse
import threading
import time
import os

import cv2
import lgpio
from picamera2 import Picamera2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoConfig
import supervision as sv
from types import SimpleNamespace

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pyserial not installed. Install with: python -m pip install pyserial"
    ) from exc

from arduino_servo_serial_toggle import (
    ANGLE_A,
    ANGLE_B,
    DEFAULT_SERVOS,
    _normalize_angles,
    send_angles,
    smooth_move,
)
from ultrasonic_distance import (
    ECHO_GPIO,
    MAX_DISTANCE_CM,
    READ_INTERVAL_SEC,
    SPEED_OF_SOUND_CM_PER_SEC,
    TRIG_GPIO,
    measure_distance_cm,
)

# ------------------------------ Configuration ------------------------------
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200
MODEL_DIR = "../rtdetr-finetuned-final"          # Directory containing RT-DETR model & rtdetr_traced.pt
CONFIDENCE = 0.5
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640
DETECTION_LINE_ORIENTATION = "horizontal"
DETECTION_LINE_POSITION = 140

DISTANCE_THRESHOLD_CM = 16.0
SECONDS_PER_CLOSE_FRAME = 0.1
REQUIRED_DETECTION_FRAMES = 3
REQUIRED_CLEAR_FRAMES = 3
REQUIRED_CAMERA_FRAMES = 1
STARTUP_HOLD_SEC = 2.0

METAL_SERVO_ID = 1
PLASTIC_SERVO_ID = 2
PAPER_SERVO_ID = 3
METAL_OPEN_DELAY_SEC = 10.4
PLASTIC_OPEN_DELAY_SEC = 19.8
PAPER_OPEN_DELAY_SEC = 29.4

METAL_CLASS_NAMES = {"metal", "can", "aluminium", "aluminum", "tin", "metal can"}
PLASTIC_CLASS_NAMES = {"plastic", "plastic bottle", "bottle plastic"}
PAPER_CLASS_NAMES = {"paper", "cardboard", "carton", "paperboard"}

# ------------------------------ Helper functions ------------------------------
def is_close_detection(distance_cm, threshold_cm):
    return distance_cm is not None and distance_cm < threshold_cm

def read_and_print_distance(handle, timeout_sec):
    distance_cm = measure_distance_cm(handle, timeout_sec)
    if distance_cm is None:
        print("Out of range / timeout")
    else:
        print(f"Distance: {distance_cm:.1f} cm")
    return distance_cm

def normalize_class_name(name):
    return name.strip().lower().replace("_", " ").replace("-", " ")

NORMALIZED_METAL_CLASS_NAMES = {normalize_class_name(n) for n in METAL_CLASS_NAMES}
NORMALIZED_PLASTIC_CLASS_NAMES = {normalize_class_name(n) for n in PLASTIC_CLASS_NAMES}
NORMALIZED_PAPER_CLASS_NAMES = {normalize_class_name(n) for n in PAPER_CLASS_NAMES}

def clamp(value, low, high):
    return max(low, min(high, value))

def box_touches_line_xyxy(xyxy, line_orientation, line_position):
    """Check if a bounding box (x1,y1,x2,y2) touches the given line."""
    x1, y1, x2, y2 = xyxy
    if line_orientation == "horizontal":
        return y1 <= line_position <= y2
    else:
        return x1 <= line_position <= x2

def draw_detection_line(image, line_orientation, line_position):
    height, width = image.shape[:2]
    if line_orientation == "horizontal":
        line_position = int(clamp(line_position, 0, height - 1))
        start = (0, line_position)
        end = (width - 1, line_position)
        label_position = (8, max(20, line_position - 8))
        label = f"line y={line_position}"
    else:
        line_position = int(clamp(line_position, 0, width - 1))
        start = (line_position, 0)
        end = (line_position, height - 1)
        label_position = (min(width - 115, line_position + 8), 24)
        label = f"line x={line_position}"

    cv2.line(image, start, end, (0, 255, 255), 2)
    cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

def best_material(material_frame_counts, confirmed_materials):
    seen = [
        (material_frame_counts[material], material)
        for material in confirmed_materials
        if material_frame_counts[material] > 0
    ]
    if not seen:
        return None
    return max(seen)[1]

def queue_summary(pending_open_events, material_actions):
    counts = {material: 0 for material in material_actions}
    for _, material, _, _ in pending_open_events:
        counts[material] += 1
    return ", ".join(
        f"{material}={counts[material]}" for material in material_actions
    )

def move_one_servo(ser, current_angles, servo_idx, target_angle, servos, servo_ids):
    target_angles = current_angles[:]
    target_angles[servo_idx] = target_angle
    if target_angles == current_angles:
        return current_angles
    return smooth_move(
        ser,
        current_angles,
        target_angles,
        servos=servos,
        servo_ids=servo_ids,
    )

# ------------------------------ Ultrasonic worker ------------------------------
def ultrasonic_worker(
    handle,
    timeout_sec,
    threshold_cm,
    required_frames,
    clear_frames_required,
    state,
    state_lock,
    stop_event,
):
    close_frames = 0
    clear_frames = 0
    confirmed_object = False
    confirmation_time = None
    active_object_id = None
    next_object_id = 1

    while not stop_event.is_set():
        now = time.monotonic()
        distance_cm = read_and_print_distance(handle, timeout_sec)
        detected = is_close_detection(distance_cm, threshold_cm)
        ended_event = None

        if detected:
            if close_frames == 0 and active_object_id is None:
                active_object_id = next_object_id
                next_object_id += 1

            close_frames += 1
            clear_frames = 0

            if close_frames >= required_frames and not confirmed_object:
                confirmed_object = True
                confirmation_time = now
                print(
                    f"Object {active_object_id} confirmed at "
                    f"{close_frames} ultrasonic close frames. Delay timer started."
                )

            print(
                f"Ultrasonic close frames: {close_frames}"
                f"{' (confirmed)' if confirmed_object else ''}"
            )
        else:
            if confirmed_object:
                clear_frames += 1
                print(f"Ultrasonic clear frames: {clear_frames}/{clear_frames_required}")

                if clear_frames >= clear_frames_required:
                    ended_event = {
                        "object_id": active_object_id,
                        "close_frames": close_frames,
                        "confirmation_time": confirmation_time,
                    }
                    close_frames = 0
                    clear_frames = 0
                    confirmed_object = False
                    confirmation_time = None
                    active_object_id = None
            else:
                close_frames = 0
                clear_frames = 0
                active_object_id = None

        with state_lock:
            state["latest_distance_cm"] = distance_cm
            state["distance_detected"] = detected
            state["close_frames"] = close_frames
            state["clear_frames"] = clear_frames
            state["confirmed_object"] = confirmed_object
            state["confirmation_time"] = confirmation_time
            state["active_object_id"] = active_object_id
            if ended_event is not None:
                state["ended_events"].append(ended_event)

        stop_event.wait(READ_INTERVAL_SEC)

# ------------------------------ RT-DETR inference & annotation ------------------------------
def run_rtdetr_inference(
    picam2,
    processor,
    rtdetr_model,
    id2label,
    conf_thresh,
    line_orientation,
    line_position,
    device,
):
    """Capture frame, run RT-DETR, filter boxes touching the line, annotate and display."""
    frame = picam2.capture_array()
    frame = frame[:, :, :3]                     # remove alpha if any (RGB)
    original_h, original_w = frame.shape[:2]

    # Preprocess
    inputs = processor(images=frame, return_tensors="pt", size={"height": 640, "width": 640})
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits, pred_boxes = rtdetr_model(inputs["pixel_values"])

    outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)
    target_sizes = torch.tensor([[original_h, original_w]]).to(device)
    detections_dict = processor.post_process_object_detection(
        outputs, threshold=conf_thresh, target_sizes=target_sizes
    )[0]

    detections = sv.Detections.from_transformers(detections_dict)
    detections = detections.with_nms(threshold=0.5, class_agnostic=False)

    # Determine materials from boxes that touch the line
    materials = set()
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        cls = detections.class_id[i]
        if not box_touches_line_xyxy(xyxy, line_orientation, line_position):
            continue
        label = id2label.get(cls, f"class_{cls}")
        norm_label = normalize_class_name(label)
        if norm_label in NORMALIZED_METAL_CLASS_NAMES:
            materials.add("metal")
        if norm_label in NORMALIZED_PLASTIC_CLASS_NAMES:
            materials.add("plastic")
        if norm_label in NORMALIZED_PAPER_CLASS_NAMES:
            materials.add("paper")

    # Annotate frame
    annotated = frame.copy()
    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        conf = detections.confidence[i]
        cls = detections.class_id[i]
        label = id2label.get(cls, f"class_{cls}")
        text = f"{label} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    draw_detection_line(annotated, line_orientation, line_position)
    cv2.imshow("Final Test Camera", annotated)
    return materials

# ------------------------------ Main -----------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use RT-DETR camera material detection and ultrasonic frame counting to open "
            "the matching servo after the configured delay."
        )
    )
    # RT-DETR specific arguments
    parser.add_argument("--model", default=MODEL_DIR,
                        help="Directory containing RT-DETR finetuned model (config, preprocessor_config) and rtdetr_traced.pt")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help="Confidence threshold for detections")
    # General arguments (unchanged)
    parser.add_argument("--port", default=ARDUINO_PORT)
    parser.add_argument("--baud", type=int, default=ARDUINO_BAUD)
    parser.add_argument("--threshold-cm", type=float, default=DISTANCE_THRESHOLD_CM)
    parser.add_argument("--seconds-per-frame", type=float, default=SECONDS_PER_CLOSE_FRAME)
    parser.add_argument("--metal-open-delay", type=float, default=METAL_OPEN_DELAY_SEC)
    parser.add_argument("--plastic-open-delay", type=float, default=PLASTIC_OPEN_DELAY_SEC)
    parser.add_argument("--paper-open-delay", type=float, default=PAPER_OPEN_DELAY_SEC)
    parser.add_argument("--startup-hold", type=float, default=STARTUP_HOLD_SEC)
    parser.add_argument("--required-frames", type=int, default=REQUIRED_DETECTION_FRAMES)
    parser.add_argument("--clear-frames", type=int, default=REQUIRED_CLEAR_FRAMES)
    parser.add_argument("--camera-frames", type=int, default=REQUIRED_CAMERA_FRAMES)
    parser.add_argument("--metal-servo-id", type=int, default=METAL_SERVO_ID)
    parser.add_argument("--plastic-servo-id", type=int, default=PLASTIC_SERVO_ID)
    parser.add_argument("--paper-servo-id", type=int, default=PAPER_SERVO_ID)
    parser.add_argument("--servos", type=int, default=DEFAULT_SERVOS)
    parser.add_argument(
        "--servo-ids",
        type=int,
        nargs="+",
        default=list(range(DEFAULT_SERVOS)),
        help="Servo IDs to address (default: 0 1 2 3).",
    )
    parser.add_argument("--angle-a", type=int, nargs="+", default=ANGLE_A)
    parser.add_argument("--angle-b", type=int, nargs="+", default=ANGLE_B)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument(
        "--line-orientation",
        choices=("horizontal", "vertical"),
        default=DETECTION_LINE_ORIENTATION,
        help="Only count boxes touching this horizontal or vertical line.",
    )
    parser.add_argument(
        "--line-position",
        type=int,
        default=DETECTION_LINE_POSITION,
        help="Line y-position for horizontal, or x-position for vertical.",
    )
    args = parser.parse_args()

    # ------------------------------ RT-DETR setup ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = args.model
    traced_model_path = os.path.join(model_dir, "rtdetr_traced.pt")
    if not os.path.exists(traced_model_path):
        raise SystemExit(f"Traced model not found at {traced_model_path}")

    print(f"Loading RT-DETR processor and config from {model_dir} ...")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    id2label = config.id2label

    print(f"Loading traced RT-DETR model from {traced_model_path} ...")
    rtdetr_model = torch.jit.load(traced_model_path, map_location=device)
    rtdetr_model.eval()

    # ------------------------------ Servo & ultrasonic setup ------------------------------
    servos = int(args.servos)
    servo_ids = args.servo_ids
    if len(servo_ids) == 1 and servos > 1:
        servo_ids = servo_ids * servos
    if len(servo_ids) != servos:
        raise SystemExit(
            f"--servo-ids must have exactly {servos} values (got {len(servo_ids)})."
        )

    for servo_id_arg, servo_id in (
        ("--metal-servo-id", args.metal_servo_id),
        ("--plastic-servo-id", args.plastic_servo_id),
        ("--paper-servo-id", args.paper_servo_id),
    ):
        if servo_id not in servo_ids:
            raise SystemExit(f"{servo_id_arg} {servo_id} is not in --servo-ids {servo_ids}.")

    closed_angles = _normalize_angles(args.angle_a, servos)
    open_angles = _normalize_angles(args.angle_b, servos)
    material_actions = {
        "metal": {
            "servo_id": args.metal_servo_id,
            "servo_idx": servo_ids.index(args.metal_servo_id),
            "open_delay": args.metal_open_delay,
        },
        "plastic": {
            "servo_id": args.plastic_servo_id,
            "servo_idx": servo_ids.index(args.plastic_servo_id),
            "open_delay": args.plastic_open_delay,
        },
        "paper": {
            "servo_id": args.paper_servo_id,
            "servo_idx": servo_ids.index(args.paper_servo_id),
            "open_delay": args.paper_open_delay,
        },
    }

    max_round_trip_sec = (MAX_DISTANCE_CM * 2.0) / SPEED_OF_SOUND_CM_PER_SEC
    timeout_sec = max_round_trip_sec * 1.5

    # Ultrasonic GPIO
    handle = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(handle, TRIG_GPIO, 0)
    lgpio.gpio_claim_input(handle, ECHO_GPIO)

    # Camera
    picam2 = Picamera2()
    config_cam = picam2.create_preview_configuration(
        main={"size": (args.camera_width, args.camera_height)}
    )
    picam2.configure(config_cam)
    picam2.start()

    # Initial servo positions
    current_angles = closed_angles[:]

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        time.sleep(2.0)
        send_angles(ser, closed_angles, servos=servos, servo_ids=servo_ids)
        time.sleep(args.startup_hold)

        print(
            f"Final test running. Ultrasonic confirms after {args.required_frames} "
            f"frames < {args.threshold_cm:.1f} cm and ends after {args.clear_frames} "
            f"clear frames. Camera confirms material after {args.camera_frames} "
            "separate frame(s)."
        )
        print(
            f"Metal -> servo {args.metal_servo_id} after {args.metal_open_delay:.1f}s; "
            f"plastic -> servo {args.plastic_servo_id} after "
            f"{args.plastic_open_delay:.1f}s; paper/cardboard -> servo "
            f"{args.paper_servo_id} after {args.paper_open_delay:.1f}s."
        )
        print(
            f"Open duration = {args.seconds_per_frame:.2f}s * close frame count. "
            "Press q in the camera window, or Ctrl+C in the terminal, to stop."
        )
        print(
            f"Camera only counts boxes touching the {args.line_orientation} "
            f"line at {args.line_position}."
        )

        # Ultrasonic thread
        ultrasonic_state = {
            "latest_distance_cm": None,
            "distance_detected": False,
            "close_frames": 0,
            "clear_frames": 0,
            "confirmed_object": False,
            "confirmation_time": None,
            "active_object_id": None,
            "ended_events": [],
        }
        ultrasonic_lock = threading.Lock()
        ultrasonic_stop = threading.Event()
        ultrasonic_thread = threading.Thread(
            target=ultrasonic_worker,
            args=(
                handle,
                timeout_sec,
                args.threshold_cm,
                args.required_frames,
                args.clear_frames,
                ultrasonic_state,
                ultrasonic_lock,
                ultrasonic_stop,
            ),
            daemon=True,
        )
        ultrasonic_thread.start()

        # Main loop state
        camera_frame_counts = {}
        camera_consecutive_frames = {}
        camera_confirmed_materials = {}
        pending_open_events = []
        servo_open_until = {}

        try:
            while True:
                now = time.monotonic()

                # Run RT-DETR inference
                camera_materials = run_rtdetr_inference(
                    picam2=picam2,
                    processor=processor,
                    rtdetr_model=rtdetr_model,
                    id2label=id2label,
                    conf_thresh=args.conf,
                    line_orientation=args.line_orientation,
                    line_position=args.line_position,
                    device=device,
                )

                # Read ultrasonic state
                with ultrasonic_lock:
                    active_object_id = ultrasonic_state["active_object_id"]
                    close_frames = ultrasonic_state["close_frames"]
                    clear_frames = ultrasonic_state["clear_frames"]
                    confirmed_object = ultrasonic_state["confirmed_object"]
                    ended_events = ultrasonic_state["ended_events"][:]
                    ultrasonic_state["ended_events"].clear()

                # Update camera frame counts for current active object
                if active_object_id is not None:
                    if active_object_id not in camera_frame_counts:
                        camera_frame_counts[active_object_id] = {
                            material: 0 for material in material_actions
                        }
                        camera_consecutive_frames[active_object_id] = {
                            material: 0 for material in material_actions
                        }
                        camera_confirmed_materials[active_object_id] = set()

                    for material in material_actions:
                        if material in camera_materials:
                            camera_frame_counts[active_object_id][material] += 1
                            camera_consecutive_frames[active_object_id][material] += 1
                        else:
                            camera_consecutive_frames[active_object_id][material] = 0

                        if (
                            camera_consecutive_frames[active_object_id][material]
                            >= args.camera_frames
                        ):
                            camera_confirmed_materials[active_object_id].add(material)

                # Print status
                print(
                    "Camera detected: "
                    f"{', '.join(sorted(camera_materials)) if camera_materials else 'none'}"
                )
                if active_object_id is None:
                    print("Camera frames: no active ultrasonic object")
                else:
                    print(
                        f"Camera frames for object {active_object_id}: "
                        + ", ".join(
                            f"{material}={camera_frame_counts[active_object_id][material]}"
                            f"{' confirmed' if material in camera_confirmed_materials[active_object_id] else ''}"
                            for material in material_actions
                        )
                    )
                print(
                    f"Ultrasonic snapshot: close={close_frames}, clear={clear_frames}, "
                    f"confirmed={'yes' if confirmed_object else 'no'}"
                )
                print(f"Queued: {queue_summary(pending_open_events, material_actions)}")

                # Process ended ultrasonic events
                for ended_event in ended_events:
                    object_id = ended_event["object_id"]
                    event_close_frames = ended_event["close_frames"]
                    confirmation_time = ended_event["confirmation_time"]
                    counts = camera_frame_counts.get(
                        object_id, {material: 0 for material in material_actions}
                    )
                    confirmed_materials = camera_confirmed_materials.get(object_id, set())
                    material = best_material(counts, confirmed_materials)

                    if material is None:
                        print(
                            f"Object {object_id} ended after {event_close_frames} "
                            "ultrasonic close frames, but camera did not confirm a "
                            "supported material."
                        )
                    else:
                        action = material_actions[material]
                        open_hold_sec = event_close_frames * args.seconds_per_frame
                        open_time = confirmation_time + action["open_delay"]
                        pending_open_events.append(
                            (open_time, material, open_hold_sec, event_close_frames)
                        )
                        pending_open_events.sort()
                        remaining = max(0.0, open_time - now)
                        print(
                            f"Object {object_id} ended after {event_close_frames} "
                            f"ultrasonic close frames as {material}. Queued servo "
                            f"{action['servo_id']} in {remaining:.1f}s for "
                            f"{open_hold_sec:.2f}s. Queued: "
                            f"{queue_summary(pending_open_events, material_actions)}."
                        )

                    # Clean up camera data for this object
                    camera_frame_counts.pop(object_id, None)
                    camera_consecutive_frames.pop(object_id, None)
                    camera_confirmed_materials.pop(object_id, None)

                # Open servos whose delay has elapsed
                while pending_open_events and pending_open_events[0][0] <= now:
                    _, material, open_hold_sec, event_frames = pending_open_events.pop(0)
                    action = material_actions[material]
                    servo_id = action["servo_id"]
                    servo_idx = action["servo_idx"]
                    current_angles = move_one_servo(
                        ser,
                        current_angles,
                        servo_idx,
                        open_angles[servo_idx],
                        servos,
                        servo_ids,
                    )
                    print(
                        f"{material.title()} servo {servo_id} opened for "
                        f"{open_hold_sec:.2f}s from {event_frames} close frames. "
                        f"Queued: {queue_summary(pending_open_events, material_actions)}."
                    )
                    servo_open_until[servo_id] = max(
                        servo_open_until.get(servo_id, 0.0),
                        time.monotonic() + open_hold_sec,
                    )

                # Close servos after their open duration
                for servo_id, open_until in list(servo_open_until.items()):
                    if time.monotonic() < open_until:
                        continue
                    servo_idx = servo_ids.index(servo_id)
                    current_angles = move_one_servo(
                        ser,
                        current_angles,
                        servo_idx,
                        closed_angles[servo_idx],
                        servos,
                        servo_ids,
                    )
                    print(f"Servo {servo_id} closed.")
                    del servo_open_until[servo_id]

                # Quit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(READ_INTERVAL_SEC)

        except KeyboardInterrupt:
            pass
        finally:
            ultrasonic_stop.set()
            ultrasonic_thread.join(timeout=1.0)
            smooth_move(
                ser, current_angles, closed_angles, servos=servos, servo_ids=servo_ids
            )
            lgpio.gpio_write(handle, TRIG_GPIO, 0)
            lgpio.gpiochip_close(handle)
            picam2.stop()
            cv2.destroyAllWindows()
            print("Stopped cleanly")

if __name__ == "__main__":
    main()