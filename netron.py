import torch

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/champ/yolov5/runs/train/ppe_detection_m_896_v112/weights/best.pt')

# Print model architecture
print(model)
