```markdown
# YOLOv11 Posture Detection on Kria KV260

Real-time **YOLOv11n** posture classification (**Good / Bad**) optimized for the **AMD Kria KV260** Vision AI Starter Kit using Vitis AI 3.5 and the DPUCZDX8G accelerator.

This project demonstrates complete model surgery, INT8 quantization, and deployment of a modern YOLO architecture on an **Arm-powered edge platform** (quad-core Cortex-A53 + DPU).

---

## Key Results

| Metric                    | Value                          |
|---------------------------|--------------------------------|
| Model                     | YOLOv11n (2-class, custom)     |
| Input Resolution          | 512 × 512                      |
| DPU Subgraphs             | 1 (fully accelerated)          |
| Inference Time (DPU)      | ~12.7 ms                       |
| Achieved Confidence       | 0.73 (Good class)              |
| Platform                  | Kria KV260 (Arm Cortex-A53)    |

---

## What Was Optimized

- **Model Surgery** for DPU compatibility  
  - SiLU → Hardsigmoid  
  - `torch.chunk` → `torch.split`  
  - Softmax approximations in Attention & DFL  
  - Simplified Head (raw outputs only)

- **INT8 Quantization** with Vitis AI 3.5 (`vai_q_pytorch`)

- **Compilation** to a single DPU subgraph targeting DPUCZDX8G (KV260)

- Custom post-processing (DFL decode + NMS) running on the Arm CPU

---

## Hardware

- **Board**: AMD Kria KV260 Vision AI Starter Kit  
- **CPU**: Quad-core Arm Cortex-A53  
- **Accelerator**: DPUCZDX8G  
- **OS**: Petalinux / Kria base image with Vitis AI Runtime

---

## Project Structure

```
├── nets/nn.py              # Surgically modified YOLOv11
├── utils/                  # Dataset, loss, and utility code
├── main.py                 # Training & evaluation entry point
├── board_inference/        # Runtime scripts for KV260
├── docs/                   # Technical process report
└── results/                # Sample detection outputs
```

---

## How to Reproduce

### 1. Training (Host PC)
```bash
python -m venv yolov11-env
source yolov11-env/bin/activate
pip install torch torchvision pyyaml tqdm opencv-python matplotlib thop
python main.py --train --batch-size 8 --epochs 60 --input-size 512
```

### 2. Quantization (Vitis AI 3.5 Docker)
```bash
# Inside vitis-ai-pytorch container
python quantize_simple.py
```

### 3. Compile for KV260
```bash
vai_c_xir -x quantize_result/YOLO_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compile_result -n yolov11_balanced
```

### 4. Run on Kria KV260
```bash
python3 run_inference.py
```

---

## Challenges Solved

- PyTorch 2.6 `weights_only` loading errors  
- Channel mismatch after CSPModule surgery  
- Quantizer tracing failures (complex Head)  
- Class scores collapsing to near-zero on DPU (fixed by input normalization `img - 128`)  
- Loose bounding boxes after INT8 quantization of DFL head  

Full technical write-up is available in `docs/YOLOv11_KV260_Process_Report.docx`.

---

## License

MIT License

---

## Author

Built for the **Arm AI Optimization Challenge 2026** – Physical AI Track.
```
