from utils.Embedder import FaceEmbedder, IMAGE_EXTS
import os
import faiss
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

if __name__ == "__main__":

    gallery_path = "E:\\Photos\\UCF Graduation"
    fe = FaceEmbedder()
    cache = fe.build_embedding_cache(gallery_path, load_pickle=True)  # if false will create a new cache

    cache_features = []
    cache_filenames = []

    for k, v in cache.items():
        if v is not None:
            for i, emb in enumerate(v):
                cache_features.append(emb["embedding"])
                cache_filenames.append(k)
                # print(k, i+1)

    
    cache_features_np = np.array(cache_features).astype('float32')
    print("Cache features shape:", cache_features_np.shape)  # (858, 512)

    image_paths = [
                os.path.join(r, f)
                for r, _, fs in os.walk(gallery_path)
                for f in fs if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    # Query image
    relpath, query_emb =  fe.encode_file("E:\\Photos\\UCF Graduation\\100CANON-20250506T025251Z-1-001\\100CANON\\IMG_5369.JPG", gallery_path)

    # Load data
    # cache_features_np = cache_features_np.astype('float32')  # (858, 512)
    query_emb_feat = query_emb[0]["embedding"].reshape(1, -1).astype('float32')  # (1, 512)
    print("Query features shape:", query_emb_feat.shape)  # (1, 512)

    # Normalize for cosine similarity
    faiss.normalize_L2(cache_features_np)
    faiss.normalize_L2(query_emb_feat)

    # Create FAISS index
    index = faiss.IndexFlatIP(cache_features_np.shape[1])  # Cosine similarity = inner product after normalization
    index.add(cache_features_np)

    # Compute similarities with all items (k = number of gallery samples)
    similarities, indices = index.search(query_emb_feat, cache_features_np.shape[0])  # returns all items

    # Define threshold for "match"
    threshold = 0.1
    matched_indices = indices[0][similarities[0] >= threshold]
    matched_scores = similarities[0][similarities[0] >= threshold]

    print("Matched indices:", matched_indices)
    print("Matched similarity scores:", matched_scores)

    matched_indices = matched_indices.tolist()

    for mat in matched_indices:
        img = Image.open(os.path.join(gallery_path, cache_filenames[mat])).convert('RGB')
        img = ImageOps.exif_transpose(img)

        filename = os.path.basename(cache_filenames[mat])

        # Copy the image to a new location
        new_path = os.path.join(gallery_path,"match")
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        new_path = os.path.join(new_path, filename)
        try:
            img.save(new_path)
            print(f"Image saved to {new_path}")
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
            continue

    # Display the image


    






        


