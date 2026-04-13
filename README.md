# Object Aware RoomReIdentification

This repository contains a PyTorch reimplementation of the AirRoom pipeline for indoor room re-identification using global, object-level, and fine-grained matching cues.

## Project Overview

The pipeline follows a coarse-to-fine retrieval strategy:

1. **Reference preprocessing**
   - Select reference images for each room
   - Segment objects from reference images
   - Extract object, patch, and room-level embeddings
   - Save these embeddings for retrieval

2. **Inference**
   - Extract a global feature for the query image
   - Retrieve top candidate rooms using global similarity
   - Re-score candidates using object and patch features
   - Use fine matching between query and reference images for final selection

---

## Repository Structure

- `preprocess.py` — builds the reference database
- `inference.py` — runs coarse-to-fine room re-identification
- `models/build_model.py` — model builders and wrappers
- `data/query_dataset.py` — query dataset loader
- `data/reference_dataset.py` — reference dataset loader
- `utils/geometry.py` — geometry utilities for receptive field expansion
- `utils/scoring.py` — scoring functions for patch/object matching
- `config/preprocess.yaml` — preprocessing configuration
- `config/inference.yaml` — inference configuration

---

## Dataset Layout

Expected dataset structure:

```text
datasets/
  ReplicaReID/
    apartment_0/
      kitchen/
        rgb/
        depth/
      living/
        rgb/
        depth/
    apartment_1/
    apartment_2/
    office_0/
    office_1/
    ...
    room_label.txt
---
```
## Installation

### 1. Clone the repository in Google Colab

```python
%cd /content
!rm -rf cv_project
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Priya-Kumari-Chourasia/cv_project.git
%cd /content/cv_project
```

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Copy dataset into the project folder

```python
%cd /content/cv_project
!rm -rf /content/cv_project/datasets/ReplicaReID
!mkdir -p /content/cv_project/datasets
!cp -r "/content/drive/MyDrive/datasets/ReplicaReID" /content/cv_project/datasets/
```

### 4. Install requirements

```python
!pip install -r requirements.txt
```

### 5. Install LightGlue

```python
%cd /content
!rm -rf LightGlue
!git clone https://github.com/cvg/LightGlue.git
%cd /content/LightGlue
!python -m pip install -e .
%cd /content/cv_project
```
### 6. Patch build_model.py for CLIP selector

```python
from pathlib import Path
import re

path = Path("/content/cv_project/models/build_model.py")
text = path.read_text()

pattern = r'''class CLIPSelector:.*?class SegmentationWrapper:'''
replacement = '''class CLIPSelector:
    def __init__(self, device: torch.device = DEVICE) -> None:
        self.device = device
        self.kind = "none"
        self.processor = None
        self.model = None
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            self.kind = "clip"
        except Exception:
            pass

    @torch.no_grad()
    def encode_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.kind == "clip" and self.model is not None and self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feat = self.model.get_image_features(**inputs)

            if hasattr(feat, "image_embeds"):
                feat = feat.image_embeds
            elif hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            elif hasattr(feat, "last_hidden_state"):
                feat = feat.last_hidden_state.mean(dim=1)

            if not torch.is_tensor(feat):
                raise TypeError(f"Unexpected CLIP feature type: {type(feat)}")

            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            return F.normalize(feat, dim=1).squeeze(0).cpu()

        tensor = transforms.ToTensor()(image)
        feat = torch.cat([tensor.mean(dim=(1, 2)), tensor.std(dim=(1, 2))], dim=0).unsqueeze(0)
        return F.normalize(feat, dim=1).squeeze(0).cpu()


class SegmentationWrapper:'''

new_text, n = re.subn(pattern, replacement, text, flags=re.S)
if n != 1:
    raise ValueError("Could not patch CLIPSelector block cleanly")

path.write_text(new_text)
print("Patched build_model.py")
```

### 7. Run preprocessing once

```python
%cd /content/cv_project
!python preprocess.py
```

### 8. Patch fine matcher block
```python
from pathlib import Path
import re

path = Path("/content/cv_project/models/build_model.py")
text = path.read_text()

old = r'''    @torch.no_grad\(\)
    def count_matches\(self, img0: Image.Image, img1: Image.Image\) -> int:
        if self.kind == "lightglue":
            from torchvision.transforms.functional import to_tensor

            im0 = to_tensor\(img0.convert\("RGB"\)\).to\(self.device\)
            im1 = to_tensor\(img1.convert\("RGB"\)\).to\(self.device\)
            feats0 = self.extractor.extract\(im0\)
            feats1 = self.extractor.extract\(im1\)
            matches = self.matcher\(\{"image0": feats0, "image1": feats1\}\)
            ms = matches\["matches"\]
            return int\(ms.shape\[0\]\)

        import cv2
        import numpy as np

        a = np.array\(img0.convert\("L"\)\)
        b = np.array\(img1.convert\("L"\)\)
        orb = cv2.ORB_create\(2000\)
        kp1, des1 = orb.detectAndCompute\(a, None\)
        kp2, des2 = orb.detectAndCompute\(b, None\)
        if des1 is None or des2 is None:
            return 0
        bf = cv2.BFMatcher\(cv2.NORM_HAMMING, crossCheck=True\)
        matches = bf.match\(des1, des2\)
        return int\(len\(matches\)\)'''

new = '''    @torch.no_grad()
    def count_matches(self, img0: Image.Image, img1: Image.Image) -> int:
        if self.kind == "lightglue":
            from torchvision.transforms.functional import to_tensor
            import torch

            im0 = to_tensor(img0.convert("RGB")).to(self.device)
            im1 = to_tensor(img1.convert("RGB")).to(self.device)

            feats0 = self.extractor.extract(im0)
            feats1 = self.extractor.extract(im1)
            out = self.matcher({"image0": feats0, "image1": feats1})

            ms = out.get("matches", None)

            if ms is None:
                return 0
            if isinstance(ms, torch.Tensor):
                return int(ms.shape[0])
            if isinstance(ms, (list, tuple)):
                return int(len(ms))

            try:
                return int(ms.shape[0])
            except Exception:
                try:
                    return int(len(ms))
                except Exception:
                    return 0

        import cv2
        import numpy as np

        a = np.array(img0.convert("L"))
        b = np.array(img1.convert("L"))
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(a, None)
        kp2, des2 = orb.detectAndCompute(b, None)
        if des1 is None or des2 is None:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return int(len(matches))'''

new_text, n = re.subn(old, new, text, flags=re.S)
if n != 1:
    raise ValueError("Could not patch count_matches block cleanly")

path.write_text(new_text)
print("Patched FineMatcher.count_matches")
```

### 9. Run preprocessing and inference
```python

%cd /content/cv_project
!python preprocess.py
!python inference.py
```

### 10. Visualize correct and wrong predictions
```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw

from models.build_model import build_segmentation

base_dir = "/content/cv_project"

csv_candidates = [
    "/mnt/data/results(1).csv",
    "/mnt/data/results.csv",
    "/content/results(1).csv",
    "/content/results.csv",
    "/content/cv_project/outputs/results.csv",
]

csv_path = None
for p in csv_candidates:
    if Path(p).exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError("Could not find results CSV")

print("Using CSV:", csv_path)
df = pd.read_csv(csv_path)

def resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(base_dir, p)

def draw_red_box(img, thickness=6):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for t in range(thickness):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=(255, 0, 0))
    return img

def make_patch_canvas(query_img, detections, max_patches=4, patch_size=(180, 130)):
    patches = []
    for det in detections[:max_patches]:
        x, y, w, h = det["bbox"]
        crop = query_img.crop((x, y, x + w, y + h)).resize(patch_size)
        patches.append(crop)

    rows, cols = 2, 2
    canvas = Image.new("RGB", (cols * patch_size[0], rows * patch_size[1]), (255, 255, 255))
    for i, p in enumerate(patches[:4]):
        r, c = divmod(i, cols)
        canvas.paste(p, (c * patch_size[0], r * patch_size[1]))
    return canvas

def make_segmentation_overlay(query_img, detections, alpha=0.45):
    img = np.array(query_img).copy()
    overlay = img.copy()
    rng = np.random.default_rng(42)

    for det in detections:
        mask = det["segmentation"].astype(bool)
        color = rng.integers(0, 255, size=3)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        x, y, w, h = det["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return Image.fromarray(blended)

def make_keypoint_vis(query_img, max_kp=500):
    img = np.array(query_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(max_kp)
    kps = orb.detect(gray, None)

    out = img.copy()
    for i, kp in enumerate(kps[:max_kp]):
        x, y = map(int, kp.pt)
        color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.circle(out, (x, y), 1, color, -1)
    return Image.fromarray(out)

def make_database_column(paths, selected_path, thumb_size=(150, 105), gap=18):
    thumbs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(thumb_size)
        if os.path.abspath(p) == os.path.abspath(selected_path):
            img = draw_red_box(img, thickness=5)
        thumbs.append(img)

    total_h = len(thumbs) * thumb_size[1] + (len(thumbs) - 1) * gap
    canvas = Image.new("RGB", (thumb_size[0], total_h), (255, 255, 255))

    y = 0
    for img in thumbs:
        canvas.paste(img, (0, y))
        y += thumb_size[1] + gap
    return canvas

def show_one_example(row, segmenter):
    query_path = resolve_path(row["query_image"])
    selected_ref_path = resolve_path(row["reference_image"])
    ref_dir = os.path.dirname(selected_ref_path)

    database_paths = sorted([
        os.path.join(ref_dir, f)
        for f in os.listdir(ref_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ])[:4]

    query_img = Image.open(query_path).convert("RGB")
    detections = segmenter.generate(query_img, score_threshold=0.6)

    global_panel = query_img.resize((420, 280))
    patch_panel = make_patch_canvas(query_img, detections, max_patches=4, patch_size=(200, 135))
    seg_panel = make_segmentation_overlay(query_img, detections).resize((420, 280))
    kp_panel = make_keypoint_vis(query_img, max_kp=500).resize((420, 280))
    db_panel = make_database_column(database_paths, selected_ref_path, thumb_size=(150, 105), gap=18)

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.15, 1.15, 0.65],
        height_ratios=[1, 1],
        wspace=0.03,
        hspace=0.12
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axis("off")

    ax1.imshow(global_panel)
    ax1.set_title("Global Context", fontsize=20, fontweight="bold")

    ax2.imshow(patch_panel)
    ax2.set_title("Object Patches", fontsize=20, fontweight="bold")

    ax3.imshow(seg_panel)
    ax3.set_title("Object Segmentation", fontsize=20, fontweight="bold")

    ax4.imshow(kp_panel)
    ax4.set_title("Keypoints", fontsize=20, fontweight="bold")

    ax5.imshow(db_panel)
    ax5.set_title("Database", fontsize=20, fontweight="bold")

    fig.suptitle(
        f'GT={row["query_room"]} | Pred={row["pred_room"]} | Correct={row["correct"]}',
        fontsize=18,
        fontweight="bold"
    )
    fig.text(0.41, 0.05, "Query", fontsize=22, fontweight="bold")
    plt.show()

correct_df = df[df["correct"] == 1].reset_index(drop=True)
wrong_df = df[df["correct"] == 0].reset_index(drop=True)

print("Correct predictions:", len(correct_df))
print("Wrong predictions:", len(wrong_df))

segmenter = build_segmentation("maskrcnn")

for i in range(min(2, len(correct_df))):
    print(f"Showing CORRECT example {i}")
    show_one_example(correct_df.iloc[i], segmenter)

for i in range(min(2, len(wrong_df))):
    print(f"Showing WRONG example {i}")
    show_one_example(wrong_df.iloc[i], segmenter)
```
