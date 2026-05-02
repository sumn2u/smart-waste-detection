import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoConfig
import supervision as sv
import numpy as np
from types import SimpleNamespace

# -------------------------------
# 1. Load processor and config (for labels)
# -------------------------------
model_path = "../rtdetr-finetuned-final"
processor = AutoImageProcessor.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
id2label = config.id2label

# -------------------------------
# 2. Load traced TorchScript model
# -------------------------------
model = torch.jit.load("rtdetr_traced.pt")
model.eval()

# -------------------------------
# 3. Load and preprocess image
# -------------------------------
image_path = "../samples/plastic_2.jpg"

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# -------------------------------
# 4. Run inference
# -------------------------------
with torch.no_grad():
    logits, pred_boxes = model(inputs["pixel_values"])

# Wrap outputs into an object that the processor expects
outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)

# -------------------------------
# 5. Post-process detections
# -------------------------------
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
detections_dict = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

detections = sv.Detections.from_transformers(detections_dict).with_nms(threshold=0.5, class_agnostic=False)
# detections = detections.with_nms(threshold=0.5) 

# -------------------------------
# 6. PRINT ALL DETECTIONS
# -------------------------------
print(f"\nFound {len(detections)} object(s):\n")
for i, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
    label = id2label.get(cls.item(), f"class_{cls.item()}")
    print(f"{i+1}. {label} (confidence: {conf:.3f})")
    print(f"   Bounding box (x1,y1,x2,y2): {box.tolist()}\n")

# -------------------------------
# 7. Visualize (optional)
# -------------------------------
img_np = np.array(image)
labels = [
    f"{id2label.get(c.item(), str(c.item()))} {s:.2f}"
    for c, s in zip(detections.class_id, detections.confidence)
]

annotated = sv.BoxAnnotator().annotate(img_np.copy(), detections)
annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
sv.plot_image(annotated, size=(12, 12))