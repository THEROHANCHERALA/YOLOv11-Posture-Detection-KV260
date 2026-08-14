import torch
import os
from pytorch_nndct.apis import torch_quantizer
from nets import nn

ckpt = torch.load('./weights/best.pt', map_location='cpu', weights_only=False)
model = ckpt['model'].float()
model.eval()

def raw_forward(x):
    x = model.net(x)
    x = model.fpn(x)
    x = list(x)
    for i, (box, cls) in enumerate(zip(model.head.box, model.head.cls)):
        x[i] = torch.cat((box(x[i]), cls(x[i])), dim=1)
    return x

model.forward = raw_forward

dummy = torch.randn(1, 3, 512, 512)

print("Loading calibrated quantizer (test mode)...")
quantizer = torch_quantizer(
    quant_mode='test',
    module=model,
    input_args=(dummy,),
    output_dir='quantize_result',
    device=torch.device('cpu')
)
quant_model = quantizer.quant_model

# Quick forward to finalize
with torch.no_grad():
    quant_model(dummy)

print("Exporting xmodel...")
quantizer.export_xmodel(output_dir='quantize_result', deploy_check=False)
print("Done!")
