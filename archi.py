import torch
from models.yolo import Model

# Load the model from the YAML configuration
model = Model(cfg='C:/Users/champ/yolov5/models/yolov5s.yaml', ch=3, nc=80)  # ch=3 for RGB input, nc=80 classes

from torchviz import make_dot

# Dummy input (e.g., 1 image, 3 channels, 640x640 resolution)
x = torch.randn(1, 3, 640, 640)
y = model(x)  # Forward pass
graph = make_dot(y[0], params=dict(model.named_parameters()))
graph.render("yolov5_visualization", format="png")  # Saves as PNG