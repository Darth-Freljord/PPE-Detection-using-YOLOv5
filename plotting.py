import pandas as pd
import matplotlib.pyplot as plt

# Load the results.csv file
results = pd.read_csv('runs/train/exp6/results.csv')

# Strip any whitespace from column names (common issue in YOLOv5 results.csv)
results.columns = results.columns.str.strip()

# Define the metrics to plot
metrics = [
    'train/box_loss', 'train/obj_loss', 'train/cls_loss',
    'val/box_loss', 'val/obj_loss', 'val/cls_loss',
    'metrics/precision', 'metrics/recall',
    'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'
]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
axes = axes.flatten()

# Plot each metric
for i, metric in enumerate(metrics):
    axes[i].plot(results[metric], label='results')
    # Add a smoothed line (similar to the dotted line in your image)
    axes[i].plot(results[metric].rolling(window=5).mean(), label='smooth', linestyle='--')
    axes[i].set_title(metric)
    axes[i].set_xlabel('Epoch')
    axes[i].legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('runs/train/ppe_detection_m_896_v112/custom_results.png')
plt.show()