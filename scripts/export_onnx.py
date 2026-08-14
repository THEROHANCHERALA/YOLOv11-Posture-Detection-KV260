import torch
from nets import nn

# Load model
checkpoint = torch.load('weights/best.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
model.eval()
model.cpu()
model.float()
model.head.training = False

# Dummy input
dummy_input = torch.randn(1, 3, 512, 512)

print("Exporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    "yolov11_posture.onnx",
    opset_version=13,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes=None,          # keep static for better quantization
    do_constant_folding=True
)

print("ONNX export successful → yolov11_posture.onnx")
