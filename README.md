# Edge Posture AI on Kria KV260  
### From YOLOv4 Baseline to Optimized YOLOv11 on Arm DPU

Real-time **Good / Bad posture** detection for office employee health monitoring, deployed on **AMD Kria KV260** (Arm Cortex-A53 + DPUCZDX8G).

Submitted for the **Arm AI Optimization Challenge — Physical AI track**.

---

## 1. Problem

Poor sitting posture is a common workplace issue. We need an **edge** system that:

- Runs fully on-device (no cloud)
- Uses low power (&lt;10 W class board)
- Gives real-time feedback on KV260

Target platform: **Kria KV260** (Arm + FPGA DPU).

---

## 2. Approach: Baseline → Optimize

This is one project with two stages:

| Stage | Model | Role |
|-------|--------|------|
| **Baseline** | YOLOv4-Leaky | Working, measured system on KV260 |
| **Optimization** | YOLOv11n | Modern detector + DPU-oriented model surgery + INT8 |

**Baseline repository (YOLOv4):**  
https://github.com/THEROHANCHERALA/posture_detection_system  

**This repository (YOLOv11 optimization):**  
Training, surgery, quant scripts, and KV260-oriented artifacts.

---

## 3. YOLOv4 vs YOLOv11 — Comparison

| Metric | YOLOv4 (baseline) | YOLOv11n (optimized path) | What it means |
|--------|-------------------|---------------------------|---------------|
| Architecture year | 2020 | 2024 | Newer feature design |
| DPU fit | Mostly native (LeakyReLU) | Needs **model surgery** | Harder optimization problem |
| Board accuracy | **79.65% mAP**, 81.42% accuracy | Strong float metrics; INT8 on DPU | Baseline is production-proven |
| Throughput | **11.17 FPS** on KV260 | DPU inference ~**12.7 ms** (see notes) | Both real-time capable |
| Speedup vs Arm CPU | **~1,117×** | Same hardware path (DPU) | FPGA offload is the win |
| Power | &lt;9 W board class | Same platform | Edge-friendly |
| Model size (class) | Heavier YOLOv4 stack | **Nano** (~2.6M params) | Better edge candidate |
| Optimization work | Standard Vitis PTQ | SiLU→Hardsigmoid, graph fixes, INT8 | Core of *this* challenge story |

### Advantages of keeping YOLOv4 as baseline

- Fully measured FPS, power, mAP, CPU vs DPU  
- Complete health-monitoring application path  
- Stable TF/Vitis deploy flow  

### Advantages of the YOLOv11 optimization

- Modern detector architecture  
- Explicit **DPU adaptation** (not plug-and-play)  
- Smaller nano model for edge  
- Clear “optimize a non-DPU-friendly model” narrative (Arm Optimization Challenge)

### Honest tradeoff

YOLOv4 is currently the **stronger finished product** on board (accuracy + full metrics).  
YOLOv11 is the **stronger optimization story** (surgery + quantizing a modern YOLO for DPU).  
Together they show: *ship a working edge system, then push the model forward.*

---

## 4. What was optimized (YOLOv11)

1. **Model surgery for DPUCZDX8G**
   - SiLU → Hardsigmoid  
   - Adjustments so the graph is quantizable/compilable for Vitis AI  
2. **INT8 quantization** (Vitis AI / related flow)  
3. **Compile** for KV260 (`kv260_arch.json`)  
4. **On-device validation** (result images under `docs/images/`)

---

## 5. Repository layout

```text
app/           # Host / board-oriented inference helpers
  predict.py
  float_test_board.py
nets/          # YOLOv11 model (post-surgery)
  nn.py
scripts/       # Train / export / quant helpers
  main.py
  export_onnx.py
  export_xmodel.py
  quantize_*.py
hardware/
  kv260_arch.json   # DPU architecture file for vai_c_xir on Kria KV260
docs/
  images/      # Board / result screenshots
  YOLOv11_KV260_Process_Report.docx
utils/         # Dataset / NMS / training utilities
