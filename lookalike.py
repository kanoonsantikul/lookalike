import os
import shutil
import argparse
import hashlib
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

# -------------------------------
# Config & Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def set_cache_directory(cache_path):
    if cache_path:
        torch.hub.set_dir(cache_path)
        print(f"[INFO] Cache directory set to: {cache_path}")

def get_cache_paths(input_dir, cache_dir):
    dir_hash = hashlib.md5(input_dir.encode()).hexdigest()
    feature_path = os.path.join(cache_dir, f"{dir_hash}_features.npy")
    list_path = os.path.join(cache_dir, f"{dir_hash}_paths.txt")
    return feature_path, list_path

def load_dinov2_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

def extract_visual_features(image_paths, model, transform):
    features = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Extracting image features"):
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                output = model(tensor)
                features.append(output.squeeze().cpu().numpy())
            except Exception as e:
                print(f"[WARN] Failed to process {image_path}: {e}")
                features.append(np.zeros(384))  # fallback for ViT-S/14
    return np.array(features)

def sort_images_by_similarity(image_paths, features):
    if not image_paths:
        return []

    remaining = list(range(len(image_paths)))
    ordered_indices = []

    # Start with the oldest file
    timestamps = [os.path.getmtime(p) for p in image_paths]
    start_index = timestamps.index(min(timestamps))
    current_index = start_index
    ordered_indices.append(current_index)
    remaining.remove(current_index)

    while remaining:
        current_feat = features[current_index]
        next_index = min(remaining, key=lambda i: -np.dot(current_feat, features[i]) / (
            np.linalg.norm(current_feat) * np.linalg.norm(features[i]) + 1e-8))
        ordered_indices.append(next_index)
        remaining.remove(next_index)
        current_index = next_index

    return [image_paths[i] for i in ordered_indices]

def rename_and_copy_images(image_paths, output_dir, prefix, start_number):
    os.makedirs(output_dir, exist_ok=True)
    padding = len(str(len(image_paths)))
    for index, image_path in tqdm(enumerate(image_paths, start=start_number), total=len(image_paths), desc="Renaming images"):
        ext = os.path.splitext(image_path)[1].lower()
        new_name = f"{prefix}{str(index).zfill(padding)}{ext}"
        shutil.copy(image_path, os.path.join(output_dir, new_name))

def rename_and_copy_video(video_paths, output_dir, prefix, start_number):
    os.makedirs(output_dir, exist_ok=True)
    padding = len(str(len(video_paths)))
    for index, video_path in tqdm(enumerate(video_paths, start=start_number), total=len(video_paths), desc="Renaming videos"):
        ext = os.path.splitext(video_path)[1].lower()
        new_name = f"vid_{prefix}{str(index).zfill(padding)}{ext}"
        shutil.copy(video_path, os.path.join(output_dir, new_name))

def run_lookalike_pipeline(input_dir, output_dir, prefix, cache_dir, use_cache, start_number_images, start_number_videos):
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".lookalike_cache")
        print(f"[INFO] No --cache specified. Using default: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    set_cache_directory(cache_dir)

    image_paths = []
    video_paths = []
    for filename in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, filename)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(full_path)
        elif filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            video_paths.append(full_path)

    if not image_paths:
        print("[WARN] No images found.")
    if not video_paths:
        print("[WARN] No videos found.")

    if image_paths:
        model, transform = load_dinov2_model()
        feature_cache_path, path_cache_path = get_cache_paths(input_dir, cache_dir)
        cache_exists = os.path.exists(feature_cache_path) and os.path.exists(path_cache_path)

        if use_cache and cache_exists:
            print("[INFO] Loading cached features...")
            with open(path_cache_path, 'r') as f:
                cached_paths = [line.strip() for line in f.readlines()]
            if cached_paths != image_paths:
                print("[WARN] Cached image paths do not match. Recomputing...")
                cache_exists = False

        if use_cache and cache_exists:
            features = np.load(feature_cache_path)
        else:
            features = extract_visual_features(image_paths, model, transform)
            np.save(feature_cache_path, features)
            with open(path_cache_path, 'w') as f:
                f.writelines([p + '\n' for p in image_paths])

        sorted_image_paths = sort_images_by_similarity(image_paths, features)
        rename_and_copy_images(sorted_image_paths, output_dir, prefix, start_number_images)

    if video_paths:
        rename_and_copy_video(video_paths, output_dir, prefix, start_number_videos)

    print(f"[DONE] Finished organizing {len(image_paths)} images and {len(video_paths)} videos to: {output_dir}")

# -------------------------------
# CLI Entry Point
# -------------------------------
def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Lookalike: Sort similar images and rename videos.")
    parser.add_argument("--input", required=True, help="Input folder with images and videos")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--prefix", type=str, default="cluster", help="Prefix for renamed files")
    parser.add_argument("--cache", type=str, help="Path to cache directory")
    parser.add_argument("--use_cache", action="store_true", help="Use cached features if available")
    parser.add_argument("--start_number_images", type=int, default=1, help="Starting number for renamed image files")
    parser.add_argument("--start_number_videos", type=int, default=1, help="Starting number for renamed video files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_arguments()
    run_lookalike_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        cache_dir=args.cache,
        use_cache=args.use_cache,
        start_number_images=args.start_number_images,
        start_number_videos=args.start_number_videos
    )
