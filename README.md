# Face Detection Multi-Threaded Algorithms in One Place

This repository contains multiple face detection pipelines for benchmarking and integration:

1. **DLib Face Detection**
2. **OpenCV Haarcascade Face Detection**
3. **MediaPipe Face Detection**

Each detection method is wrapped in an easy-to-use interface and can be used as building blocks for more complex applications.

---

## Getting Started

### Install DLib

```bash
pip install cmake
git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py install --no DLIB_GIF_SUPPORT
```

### Install Other Requirements

```bash
pip install -r requirements.txt
```

---

# FaissFinder: AI-Powered Face Search App

FaissFinder (aka Commence Match) is an AI-powered face search app designed to find and group similar face images from large personal photo collections like graduation albums, family trips, or public events.

### Why I Built This

After receiving hundreds of graduation pictures, I realized I didn’t want to go through each image manually. Rather than using an online face grouping tool, I wanted to build something custom and educational. Having worked with FAISS for image classification and recall-based tasks, I decided to use it for face similarity search this time, powered by CLIP-style embeddings and Gradio UI.

---

## What This App Does

- Detects **people** in all images using YOLOv8n.
- For each detected person, **crops** the upper body (more head, less legs).
- Runs **face detection and embedding extraction** on cropped images using `InsightFace` (512-dim vectors).
- Creates a **database of face embeddings**.
- Accepts a **query image**, extracts its face embedding, and performs **similarity search using FAISS**.
- Displays **all similar images** (above a chosen confidence threshold) via **Gradio**.

---

## Faiss Matching Workflow

### Building the Embedding Database

```python
cache_feat_np = np.array(features).astype("float32")
faiss.normalize_L2(cache_feat_np)
index = faiss.IndexFlatIP(cache_feat_np.shape[1])  # cosine similarity
index.add(cache_feat_np)
```

### Querying for Matches

```python
query_feat = query_emb.reshape(1, -1).astype("float32")
faiss.normalize_L2(query_feat)
similarities, indices = index.search(query_feat, len(cache_feat_np))
matched_indices = indices[0][similarities[0] >= threshold]
```

- The `threshold` can be adjusted via the Gradio UI (default recommended: 0.4+).
- False positives usually increase below 0.2.

---

## Running the App

```bash
python 3. FaissMatch.py
```

You’ll be prompted to:
- Select a gallery directory
- Load or regenerate face embedding cache
- Provide a query face image
- Set confidence threshold
- View matched results in Gradio

---

## Use Cases

- Face grouping and photo sorting
- Person search in marathon or event images
- Product similarity search
- Visual defect retrieval and classification

---

## Stack Used

- **Gradio**: Lightweight UI for launching the app in browser
- **YOLOv8n**: Person detection from Ultralytics
- **InsightFace**: Face detection and state-of-the-art embedding extraction
- **FAISS**: Efficient similarity search over large embedding spaces

---

## Results

With a single query image and a confidence threshold of 0.4, the app retrieved 356 photos, all accurately containing my face. FAISS gives lightning-fast search times; the bottleneck remains image loading and rendering in Gradio.

---

## Known Issues

- Image display in Gradio for large datasets is slow; optimization is pending.
- Face embedding extraction is not fully optimized for high-res images. This process can be improved for faster indexing on large datasets.

---

## References

- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [InsightFace](https://insightface.ai/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/#performance-metrics)
- [Face Detection Zoo Repository](https://github.com/ajaymin28/face-detection-zoo)

---
