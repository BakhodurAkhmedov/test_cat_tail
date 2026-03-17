# Training Strategy

## Model

**YOLOv11n-seg** (nano variant — segmentation)
- Pretrained on COCO (80 classes, instance segmentation)
- ~2.8M parameters
- Fast inference, suitable for demo

## Transfer Learning Approach

### Backbone Freezing

Freeze the first 10 layers (backbone / feature extractor):

```python
model = YOLO("yolo11n-seg.pt")
model.train(
    data="data/dataset.yaml",
    freeze=10,  # freeze backbone layers
    ...
)
```

**Why freeze:**
- Backbone already extracts general visual features (edges, textures, shapes)
- With only ~40 images, training all layers leads to severe overfitting
- Only the detection/segmentation head needs to learn "what a tail looks like"

### If results are poor, try:
1. Unfreeze backbone, use very low learning rate (lr0=0.0001)
2. Use more aggressive augmentation
3. Add more synthetic data

## Hyperparameters

| Parameter    | Value   | Rationale                                    |
|-------------|---------|----------------------------------------------|
| `imgsz`     | 640     | Standard YOLO input; good balance            |
| `batch`     | 8       | Small dataset, small batch                   |
| `epochs`    | 100     | Small dataset needs more passes; early stop  |
| `freeze`    | 10      | Freeze backbone for transfer learning        |
| `lr0`       | 0.01    | Default YOLO LR, fine for head-only training |
| `lrf`       | 0.01    | Final LR factor (cosine decay)               |
| `patience`  | 20      | Early stopping patience                      |
| `optimizer` | AdamW   | Better for small datasets than SGD           |
| `cos_lr`    | True    | Cosine LR schedule                           |
| `close_mosaic` | 10  | Disable mosaic last 10 epochs for stability  |

## Augmentation Strategy

YOLO built-in augmentations (configured via `train()` args):

| Augmentation      | Value | Purpose                           |
|-------------------|-------|-----------------------------------|
| `hsv_h`           | 0.015 | Hue variation (fur color)         |
| `hsv_s`           | 0.7   | Saturation variation              |
| `hsv_v`           | 0.4   | Brightness variation              |
| `degrees`         | 15    | Rotation (tails at angles)        |
| `translate`       | 0.1   | Position shift                    |
| `scale`           | 0.5   | Scale variation                   |
| `fliplr`          | 0.5   | Horizontal flip                   |
| `flipud`          | 0.0   | No vertical flip (unnatural)      |
| `mosaic`          | 1.0   | Mosaic augmentation               |
| `mixup`           | 0.1   | Light mixup for regularization    |
| `erasing`         | 0.1   | Random erasing for robustness     |

## Expected Metrics

With ~40 annotated images + synthetic augmentation:

| Metric       | Optimistic | Realistic | Minimum Acceptable |
|-------------|------------|-----------|-------------------|
| mAP50 (box) | 0.75+      | 0.55-0.70 | 0.40              |
| mAP50 (mask)| 0.70+      | 0.50-0.65 | 0.35              |
| mAP50-95    | 0.45+      | 0.30-0.40 | 0.20              |

**Note:** Tails are thin, elongated objects — mAP50-95 will be lower than typical object detection because IoU thresholds >0.5 are hard to hit for thin shapes.

## Training Command

```bash
python train.py
```

Or directly:

```bash
yolo segment train \
    model=yolo11n-seg.pt \
    data=data/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    freeze=10 \
    patience=20 \
    optimizer=AdamW \
    cos_lr=True \
    project=runs \
    name=tail_seg
```

## Post-Training

1. Best weights saved to `runs/tail_seg/weights/best.pt`
2. Metrics plotted in `runs/tail_seg/` (confusion matrix, PR curve, loss curves)
3. Run `predict.py` on val set and new internet images to check generalization
4. If mAP50 < 0.40 on val → iterate on data quality or try unfreezing backbone

## Overfitting Mitigation

Small datasets are prone to overfitting. Countermeasures:

1. **Frozen backbone** — reduces trainable parameters dramatically
2. **Heavy augmentation** — effectively increases dataset size
3. **Early stopping** — patience=20, monitors val mAP
4. **Synthetic data** — adds diversity beyond real samples
5. **Weight decay** — built into AdamW optimizer
6. **Dropout** — inherent in YOLO architecture
7. **Small model** — nano variant has fewer parameters to overfit
