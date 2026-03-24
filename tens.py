import torch
from torchviz import make_dot
from models.yolo import Model

# Load the trained YOLOv5s model
model = torch.load('C:/Users/champ/yolov5/runs/train/ppe_detection_m_896_v112/weights/best.pt', map_location='cuda')['model']

# Move the model to GPU
model = model.half().cuda()

# Create a dummy input in FP16 and move it to GPU
dummy_input = torch.zeros(1, 3, 896, 896).half().cuda()

# Forward pass to get the outputs
outputs = model(dummy_input)

# Select the first output tensor (or concatenate all outputs)
if isinstance(outputs, list):
    output_tensor = outputs[0]  # Select the first output tensor
    # Alternatively, concatenate all outputs:
    # output_tensor = torch.cat(outputs, dim=1)
else:
    output_tensor = outputs

# Generate the graph
dot = make_dot(output_tensor, params=dict(model.named_parameters()))

# Save and view the graph
dot.render('yolov5s_architecture', format='png')