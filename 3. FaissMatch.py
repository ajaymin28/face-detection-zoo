import os
import io
import gradio as gr
import numpy as np
import faiss
from PIL import Image, ImageOps
import base64
from utils.Embedder import FaceEmbedder
import uuid

# Load once on startup
# gallery_path = None
fe = FaceEmbedder()
cache_features_np = None
cache_filenames = None

def load_cache(gallery_path, load_pickle, cache_file_name, pr=gr.Progress(track_tqdm=True)):
    global cache_features_np, cache_filenames,index

    # cache = fe.build_embedding_cache(gallery_path, load_pickle=load_pickle, cache_file_name=cache_file_name)

    features, filenames = [], []
    # progress_updates = []

    # def progress_callback(current, total):
    #     msg = f"Processing {current}/{total} images..."
    #     print(msg)
    #     # progress_updates.append(msg)
    #     yield msg

    
    yield "Starting cache loading..."

    cache = fe.build_embedding_cache(
        gallery_path,
        load_pickle=load_pickle,
        cache_file_name=cache_file_name,
        # progress_callback=progress_callback
    )

    for k, v in cache.items():
        if v is not None:
            for emb in v:
                features.append(emb["embedding"])
                filenames.append(k)

    if not features:
        yield "No embeddings found in gallery."
        return


    cache_features_np = np.array(features).astype("float32")
    faiss.normalize_L2(cache_features_np)
    index = faiss.IndexFlatIP(cache_features_np.shape[1])
    index.add(cache_features_np)
    cache_filenames = filenames

    yield f"Cache loaded: {len(features)} embeddings"
    return


from concurrent.futures import ThreadPoolExecutor

def load_and_prepare_image(idx):
    try:
        match_path =  os.path.join(gallery_path, cache_filenames[idx])
        img = Image.open(match_path).convert("RGB")
        img.thumbnail((512, 512))
        img = ImageOps.exif_transpose(img)
        return img
    except Exception as e:
        print(f"Error loading image {idx}: {e}")
        return None


# Helper: PIL Image → base64
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + img_str

# Core Function
def find_similar_faces(gallery_path_, image, threshold=0.1):
    uuid_str = str(uuid.uuid4())
    global gallery_path


    if cache_features_np is None or index is None:
        return [], "⚠️ Please load the embedding cache first."

    temp_path = f"temp_query_{uuid_str}.jpg"
    image.save(temp_path)

    
    gallery_path = gallery_path_
    if type(gallery_path_) == gr.File:
        gallery_path = gallery_path_.name 
    print(temp_path, gallery_path)
    
    relpath, query_emb = fe.encode_file(temp_path, gallery_path)
    if query_emb is None:
        return [], "❌ No face found in query image."

    query_feat = query_emb[0]["embedding"].reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_feat)

    similarities, indices = index.search(query_feat, len(cache_features_np))
    matched_indices = indices[0][similarities[0] >= threshold]

    if len(matched_indices) == 0:
        return [], "⚠️ No matches found above threshold."

    yield [], f"Search is done 🎯, preparing results for {len(matched_indices)} images"
    print("Search is done, preparing results...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_and_prepare_image, matched_indices))

    # Filter out failed loads
    results = [img for img in results if img is not None]

    yield [], f"Results prepared. {len(matched_indices)} images will be rendered shortly..."

    print("Results prepared. returning to Gradio...")
    yield results, f"✅ Rendered {len(results)} matches."
    return


import gradio as gr



# # Gradio UI
# iface = gr.Interface(
#     fn=find_similar_faces,
#     inputs=[
#         gr.Image(type="pil", label="Upload Query Image", height=512, width=512),
#         gr.Slider(0.0, 1.0, value=0.1, label="Similarity Threshold"),
#     ],
#     outputs=[
#         gr.Gallery(label="Matching Images", columns=4, height="auto"),
#         gr.Textbox(label="Status")
#     ],
#     title="FAISS Finder/Commence Match",
#     description="Upload a face image to find visually similar matches from a gallery using FAISS."
# )
# ---------- UI ----------

if __name__ == "__main__":
    with gr.Blocks(title="FaissFinder | Commence Match", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🔍 FaissFinder: AI-Powered Face Search (Commence Match)")
        gr.Markdown("Upload a face image to find similar faces across your personal gallery using fast FAISS search and smart embeddings.")

        # ---------- Gallery Setup ----------
        gr.Markdown("### 📁 Gallery Setup")
        with gr.Row():
            gallery_path_gadget = gr.Textbox(label="📂 Gallery Path", value="E:\\Photos\\UCF Graduation", scale=2)
            cache_file_name = gr.Textbox(label="📄 Cache File Name", value="face_embedding_cache.pkl", scale=1)
            load_pickle = gr.Checkbox(label="📦 Load Existing Cache", value=True)
        cache_status = gr.Textbox(label="📝 Status", interactive=False)

        load_button = gr.Button("🚀 Load Embedding Cache")
        load_button.click(
            load_cache,
            inputs=[gallery_path_gadget, load_pickle, cache_file_name],
            outputs=cache_status,
            show_progress=True
        )

        # ---------- Query & Results ----------
        gr.Markdown("### 🎯 Query & Results")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                query_image = gr.Image(type="pil", label="🧑 Upload Query Image", height=256, width=256)
                threshold = gr.Slider(0.0, 1.0, value=0.1, label="🎚️ Similarity Threshold", interactive=True)
                search_button = gr.Button("🔎 Find Similar Faces")
                status_text = gr.Textbox(label="Result Info", interactive=False)

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(label="🎯 Matching Images", columns=10, height="auto")

        search_button.click(
            find_similar_faces,
            inputs=[gallery_path_gadget, query_image, threshold],
            outputs=[output_gallery, status_text]
        )

    demo.launch()


# if __name__ == "__main__":
#     with gr.Blocks(title="FaissFinder/Commence Match", theme=gr.themes.Soft()) as demo:
#         gr.Markdown("## 🎯 FaissFinder: AI-Powered Face Search")
#         gr.Markdown("Upload a face image to find similar faces across your personal gallery using fast FAISS search and smart embeddings.")


#         with gr.Row():
#             gallery_path_gadget = gr.Textbox(label="Gallery Path", value="E:\\Photos\\UCF Graduation")
#             cache_file_name = gr.Textbox(label="Cache File Name", value="face_embedding_cache.pkl")
#             load_pickle = gr.Checkbox(label="Load Pickle", value=True)
#             cache_status = gr.Textbox(label="Status")

#         load_button = gr.Button("Load Cache")
#         load_button.click(load_cache, inputs=[gallery_path_gadget, load_pickle, cache_file_name], outputs=cache_status, show_progress=True)

#         with gr.Row():
#             query_image = gr.Image(type="pil", label="Upload Query Image" , height=256, width=256)
#             threshold = gr.Slider(0.0, 1.0, value=0.1, label="Similarity Threshold")
            
#         search_button = gr.Button("Find Similar Faces")
#         status_text = gr.Textbox(label="Result Info")
#         output_gallery = gr.Gallery(label="Matching Images", columns=4, height="auto")
        

#         search_button.click(find_similar_faces, inputs=[gallery_path_gadget,query_image, threshold], outputs=[output_gallery, status_text])

#     demo.launch()


    # iface.launch()
