from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8l.pt').to(device)
results = model('examples/family.jpg', save=True)
# results = model('examples/cars.jpg', save=True)
