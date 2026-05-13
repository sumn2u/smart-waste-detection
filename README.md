# Smart Waste Detection

Two waste recognition pipelines:

- **Smart Bin** – Classification (single item)
- **Conveyor Belt** – Object Detection (multiple items)

Categories: 
 - **Smart Bin** –  `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
 - **Conveyor Belt** – `cardboard`, `glass`, `metal`, `paper`, `plastic`

---

## Project Structure

```
├── smart_bin/
│   ├── classifier.py
│   ├── camera_classification.py
│   └── smart_bin.pth
├── conveyor_belt/
│   ├── detector.py
│   ├── camera_detection.py
│   └── rtdetr_traced.pt
├── requirements.txt
└── README.md

```

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS

# or

venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.13.0.90
Pillow>=9.0.0
numpy>=1.24.0
```

---

## Smart Bin (Classification)

Run the classifier:

```bash
cd smart_bin
python classifier.py
```

Edit `classifier.py` to change the model path or image:

```python
model_path = 'smart_bin.pth'
img_path = 'your_image.jpg'
```

### Output

```txt
Prediction: plastic
Confidence: 0.9874
```

---

## Conveyor Belt (Detection)

Run the detector:

```bash
cd conveyor_belt
python detector.py
```

Prints each detection with label, confidence, and bounding box.

To change the input image, edit `image_path` in `detector.py`.

### Output Example

```txt
Found 2 object(s):

1. plastic (confidence: 0.94)
   Bounding box: [120, 80, 450, 410]

2. metal (confidence: 0.87)
   Bounding box: [520, 200, 780, 495]
```

---

## License

MIT
