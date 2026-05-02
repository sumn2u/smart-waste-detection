import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def load_model(weights_path, num_classes=6):
    # Use mobilenet_v3_small (not large) to match training
    model = models.mobilenet_v3_small(weights=None)
    
    # The classifier is nn.Sequential; last layer is Linear
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    model_path = 'smart_bin.pth'   # your trained model
    img_path = '../samples/plastic_2.jpg'
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    model = load_model(model_path)
    
    # Transform must match validation transform used in training
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)   # add batch dimension
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    print(f'Prediction: {class_names[predicted_idx.item()]}')
    print(f'Confidence: {confidence.item():.4f}')

if __name__ == '__main__':
    main()