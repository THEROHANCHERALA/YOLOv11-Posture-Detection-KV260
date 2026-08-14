import torch
import cv2
import numpy as np
from nets import nn
from utils import util
import yaml

# Load class names
with open('utils/args.yaml') as f:
    params = yaml.safe_load(f)

# Load model
checkpoint = torch.load('./weights/best.pt', map_location='cuda', weights_only=False)
model = checkpoint['model'].float().cuda().eval()

# Image path (change this)
image_path = '/home/gov/Dataset/COCO/images/val2017/'   # put full image name below

# Choose one image
import os
images = sorted(os.listdir('/home/gov/Dataset/COCO/images/val2017/'))
image_path = '/home/gov/YOLOv11-pt/test_board.jpg'   # first validation image

print("Testing on:", image_path)

# Load and preprocess image
img0 = cv2.imread(image_path)
img = cv2.resize(img0, (512, 512))
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).cuda().float() / 255.0
img = img.unsqueeze(0)

# Inference
with torch.no_grad():
    outputs = model(img)
    outputs = util.non_max_suppression(outputs, confidence_threshold=0.25, iou_threshold=0.45)

# Draw boxes
names = params['names']
colors = [(0, 0, 255), (0, 255, 0)]  # red for bad, green for good

for output in outputs:
    if output is None or len(output) == 0:
        continue
    for *xyxy, conf, cls in output:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}"
        color = colors[int(cls)]
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Save result
cv2.imwrite('result.jpg', img0)
print("Result saved as result.jpg")
