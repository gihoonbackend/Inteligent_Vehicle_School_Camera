import torch

model = torch.load("yolov8n.pt", map_location='cpu')
print(model['model'].names)  # YOLOv5, YOLOv8에서 작동

