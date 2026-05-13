import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "smart_bin.pth"
# Assuming samples is one level up from the script folder
IMAGE_PATH = BASE_DIR.parent / "samples" / "paper_1.jpg"

# 2. Load optimized model
def load_lite_model(path, num_classes=5): # Updated to 5 classes based on your config
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
        
    # Build architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    # JIT trace/script makes it faster on Windows/Mac CPUs
    return torch.jit.script(model)

# 3. Preprocessing (Using torchvision transforms)
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_inference(model, img_path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")

    img = Image.open(img_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    
    # Match precision (Float vs Double fix)
    model_dtype = next(model.parameters()).dtype
    tensor = tensor.to(model_dtype)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
        
    return idx.item(), conf.item()

# Usage
if __name__ == "__main__":
    # Updated to your 5 classes from the JSON
    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    try:
        lite_model = load_lite_model(MODEL_PATH, num_classes=len(CLASSES))
        class_id, score = run_inference(lite_model, IMAGE_PATH)
        
        print(f"\n--- Classification Result ---")
        print(f"File: {IMAGE_PATH.name}")
        print(f"Prediction: {CLASSES[class_id]}")
        print(f"Confidence: {score:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")