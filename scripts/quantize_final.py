import torch
import os
import cv2
import numpy as np
from pytorch_nndct.apis import torch_quantizer
from nets import nn

# ---------- Load model ----------
ckpt = torch.load('./weights/best.pt', map_location='cpu', weights_only=False)
model = ckpt['model'].float()
model.eval()

# Force RAW outputs (list) for DPU — post-process stays on CPU
def raw_forward(x):
    x = model.net(x)
    x = model.fpn(x)
    x = list(x)
    for i, (box, cls) in enumerate(zip(model.head.box, model.head.cls)):
        x[i] = torch.cat((box(x[i]), cls(x[i])), dim=1)
    return x

model.forward = raw_forward

# ---------- Dummy input ----------
input_size = 512
dummy = torch.randn(1, 3, input_size, input_size)

print("Creating quantizer...")
quantizer = torch_quantizer(
    quant_mode='calib',
    module=model,
    input_args=(dummy,),
    output_dir='quantize_result',
    device=torch.device('cpu')
)
quant_model = quantizer.quant_model

# ---------- Calibration images ----------
img_dir = '/workspace/Dataset/COCO/images/val2017'
img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])[:50]
print(f"Calibrating with {len(img_list)} images...")

for name in img_list:
    path = os.path.join(img_dir, name)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        quant_model(img)

print("Exporting quantized model...")
quantizer.export_quant_config()
quantizer.export_xmodel(output_dir='quantize_result', deploy_check=False)
print("Done → quantize_result/")
