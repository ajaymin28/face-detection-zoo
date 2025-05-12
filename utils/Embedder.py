import os
import pickle
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

CACHE_NAME = "face_embedding_cache.pkl"
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


class FaceEmbedder:
    def __init__(self, face_detection_method="yolo",
                 pose_model_complexity: int = 1,
                 face_min_confidence: float = 0.5,
                 yolo_conf= 0.5,
                 yolo_model='yolov8n.pt',
                 max_image_size=(1024, 1024)):
        self.method = face_detection_method

        self.yolo = YOLO(yolo_model)
        self.conf = yolo_conf
        self.max_image_size = max_image_size

        # Initialize model (once)
        self.app = FaceAnalysis(name='buffalo_l')  # or 'antelopev2' for smaller model
        self.app.prepare(ctx_id=0, det_size=max_image_size)  # ctx_id=-1 for CPU


    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
            if img.width > self.max_image_size[0] or img.height > self.max_image_size[1]:
                img.thumbnail(self.max_image_size)
            return img
    
    
    def detect_persons_yolo(self, image: np.ndarray, conf=None) -> list[tuple[int, int, int, int]]:
        """Detect 'person' class from image using YOLOv8 and return bounding boxes."""
        if conf is None:
            conf = self.conf

        results = self.yolo(image, conf=conf)[0]
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # COCO class 0 = person
                x1, y1, x2, y2 = map(int, box)
                boxes.append((y1, x2, y2, x1))  # top, right, bottom, left
        return boxes

    
    def getPersonCrop(self, arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Crop the image to the bounding box of the person."""
        top, right, bottom, left = box
        H, W, _ = arr.shape
        # Box dimensions
        box_height = bottom - top
        # box_width = right - left
        # How much to shift upward (e.g., 15% of box height)
        up_shift = int(0.15 * box_height)
        # How much of the box to keep vertically (e.g., upper 60%)
        visible_height = int(0.6 * box_height)
        # Adjust top and bottom
        adjusted_top = max(0, top - up_shift)
        adjusted_bottom = min(H, adjusted_top + visible_height)
        # Adjust left/right within bounds
        adjusted_left = max(0, left)
        adjusted_right = min(W, right)
        # Final crop
        crop = arr[adjusted_top:adjusted_bottom, adjusted_left:adjusted_right]
        return crop
    
    
    def expand_face_box(self,top, right, bottom, left,img_shape,scale=1.6, shift_up_ratio=0.25):
        """
        Expand the box around center, but shift upward to include more forehead.
        
        Args:
            scale: overall box expansion factor
            shift_up_ratio: how much to shift up (0.25 = 25% of new height)
            img_shape: (H, W) for clamping to image size
        Returns:
            new_top, new_right, new_bottom, new_left
        """
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        width = (right - left)
        height = (bottom - top)

        new_width = width * scale
        new_height = height * scale

        # shift center upward
        center_y -= new_height * shift_up_ratio

        new_left = int(center_x - new_width / 2)
        new_right = int(center_x + new_width / 2)
        new_top = int(center_y - new_height / 2)
        new_bottom = int(center_y + new_height / 2)

        if img_shape:
            H, W = img_shape[:2]
            new_top = max(0, new_top)
            new_bottom = min(H, new_bottom)
            new_left = max(0, new_left)
            new_right = min(W, new_right)

        return new_top, new_right, new_bottom, new_left

    def encode_file(self, path: str, gallery_dir: str) -> tuple[str, np.ndarray] | tuple[None, None]:
        """Encode a single image file and return its relative path and encodings."""

        rel = os.path.relpath(path, gallery_dir)
        final_encodings = []
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening file {path}: {e}")
            return None, None

        img = ImageOps.exif_transpose(img)
        img = self._resize_if_needed(img)
        arr = np.array(img)

        person_boxes = self.detect_persons_yolo(arr)
            
        if len(person_boxes)==0:
            print(f"No person detected in {path}")
            return rel, final_encodings

        print(f"Found {len(person_boxes)} person boxes in {path}")
        for pbox in person_boxes:

            person_crop = self.getPersonCrop(arr, pbox)

            # InsightFace expects BGR
            bgr_face = cv2.cvtColor(person_crop, cv2.COLOR_RGB2BGR)

            # Run detection to get aligned face & embedding
            faces = self.app.get(bgr_face)

            if faces and hasattr(faces[0], 'embedding'):
                for face in faces:
                    # face attributes
                    # 'bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'
                    final_encodings.append({
                        "embedding": face.embedding,
                        "bbox": face.bbox,
                        "gender": face.gender,
                        "age": face.age,
                    })

        return rel, final_encodings  # returning embeddings found

    def build_embedding_cache(self, gallery_dir, load_pickle=False, cache_file_name=None, progress_callback=None):
        
        if cache_file_name is None:
            cache_file_name = CACHE_NAME
        
        cache = {}
        cache_path = os.path.join(gallery_dir, cache_file_name)

        if not load_pickle:
            print(f"Cache file {cache_path} does not exist. Creating a new one.")
            image_paths = [
                os.path.join(r, f)
                for r, _, fs in os.walk(gallery_dir)
                for f in fs if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]

            # cnt = 0
            for cnt_idx,p in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images"):
                rel = os.path.relpath(p, gallery_dir)
                if rel in cache:
                    print(f"[WARN] Duplicate file {rel} found. Skipping.")
                    continue

                try:
                    rel, embedings = self.encode_file(p, gallery_dir)
                    cache[rel] = embedings
                    # cnt += 1
                    if progress_callback:
                        print("##Calling callback")
                        progress_callback(cnt_idx + 1, len(image_paths))
                except Exception as e:
                    print(f"[ERROR] Failed for {rel}: {e}")

                # if cnt % 50 == 0:
                #     break

            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Cache saved to {cache_path}")
            return cache
        
        else:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except FileNotFoundError:
                    # This case should ideally be caught by os.path.exists, but good to have
                    print(f"Cache file not found: {cache_path}")
                except Exception as e:
                    # Catch other potential errors during loading
                    print(f"Error loading cache file: {e}")
                    # You might want to consider regenerating the cache here
            else:
                print(f"Cache file {cache_path} does not exist. Please set load_pickle=False to create a new one.")
                return cache