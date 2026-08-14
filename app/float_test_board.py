
import cv2
import numpy as np
import torch
from nets import nn

WEIGHTS = "weights/best.pt"
IMAGE = "test_board.jpg"
INPUT_SIZE = 512
CONF_THRESH = 0.20
IOU_THRESH = 0.45
CLASS_NAMES = ["bad", "good"]
STRIDES = [8, 16, 32]

def letterbox(img, new_shape=512):
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)

def dfl_decode(box, ch=16):
    n = box.shape[1]
    box = box.reshape(4, ch, n)
    box = box - np.max(box, axis=1, keepdims=True)
    e = np.exp(box)
    e = e / (e.sum(axis=1, keepdims=True) + 1e-6)
    return (e * np.arange(ch, dtype=np.float32).reshape(1, ch, 1)).sum(1)

def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return keep

def main():
    model = nn.yolo_v11_n(2)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt, strict=False)
    model.eval()

    img0 = cv2.imread(IMAGE)
    assert img0 is not None, IMAGE
    img, ratio, pad = letterbox(img0, INPUT_SIZE)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outs = model(x)

    all_boxes, all_scores, all_cls = [], [], []
    for i, out in enumerate(outs):
        out = out[0].cpu().numpy()
        c, h, w = out.shape
        feat = out.reshape(c, -1)
        ltrb = dfl_decode(feat[:64])
        cls = feat[64:]
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ax = gx.ravel().astype(np.float32) + 0.5
        ay = gy.ravel().astype(np.float32) + 0.5
        boxes = np.stack([
            (ax - ltrb[0]) * STRIDES[i],
            (ay - ltrb[1]) * STRIDES[i],
            (ax + ltrb[2]) * STRIDES[i],
            (ay + ltrb[3]) * STRIDES[i],
        ], axis=1)
        scores = 1.0 / (1.0 + np.exp(-np.clip(cls, -50, 50)))
        cid = scores.argmax(0)
        conf = scores[cid, np.arange(scores.shape[1])]
        mask = conf > CONF_THRESH
        if not np.any(mask):
            continue
        boxes, conf, cid = boxes[mask], conf[mask], cid[mask]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
        all_boxes.append(boxes)
        all_scores.append(conf)
        all_cls.append(cid)

    if not all_boxes:
        print("Detections: 0")
        return

    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    cids = np.concatenate(all_cls, 0)
    keep = nms(boxes, scores, IOU_THRESH)
    boxes, scores, cids = boxes[keep], scores[keep], cids[keep]
    best = int(np.argmax(scores))
    print("Detections: 1 (showing best of %d)" % len(scores))
    print("  %s: %.2f %s" % (CLASS_NAMES[int(cids[best])], float(scores[best]), list(map(int, boxes[best]))))

    x1, y1, x2, y2 = map(int, boxes[best])
    name = CLASS_NAMES[int(cids[best])]
    color = (0, 200, 0) if name == "good" else (0, 0, 220)
    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img0, "%s %.2f" % (name, scores[best]), (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite("result_float_board.jpg", img0)
    print("Saved -> result_float_board.jpg")

if __name__ == "__main__":
    main()
