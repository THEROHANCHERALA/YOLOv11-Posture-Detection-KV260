import torch
from pytorch_nndct.apis import torch_quantizer
from nets import nn

checkpoint = torch.load('weights/best.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
model.eval()
model.cpu()
model.float()

# Force the simplified path
model.head.training = False

input_tensor = torch.randn(1, 3, 512, 512)

print("Starting quantization with simplified Head...")

quantizer = torch_quantizer(
    quant_mode='calib',
    module=model,
    input_args=(input_tensor,),
    output_dir='quantize_result',
    device=torch.device('cpu')
)

quant_model = quantizer.quant_model

print("Running calibration...")
with torch.no_grad():
    for i in range(20):
        quant_model(torch.randn(1, 3, 512, 512))

quantizer.export_quant_config()
quantizer.export_xmodel(deploy_check=False)

print("Done! Check quantize_result/")
