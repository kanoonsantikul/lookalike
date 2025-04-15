import os
import shutil
import argparse
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import hashlib

import torch
import torchvision.transforms as transforms

# -------------------------------
# Config & Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------------
# Set Lookaloke Cache Directory
# -------------------------------
def set_cache_directory(cache_path):
    if cache_path:
        torch.hub.set_dir(cache_path)
        print(f"[INFO] Cache directory set to: {cache_path}")

# -------------------------------
# Retrieve Path For Features Cache
# -------------------------------
def get_cache_paths(input_dir, cache_dir):
    dir_hash = hashlib.md5(input_dir.encode()).hexdigest()
    feature_path = os.path.join(cache_dir, f"{dir_hash}_features.npy")
    list_path = os.path.join(cache_dir, f"{dir_hash}_paths.txt")
    return feature_path, list_path

# -------------------------------
# Load DINOv2 Model
# -------------------------------
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

# -------------------------------
# Extract Visual Features
# -------------------------------
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

# -------------------------------
# Dimensionality Reduction
# -------------------------------
def apply_pca(features, components=100):
    print(f"[INFO] Reducing features to {components} dimensions using PCA...")
    pca = PCA(n_components=components)
    return pca.fit_transform(features)

# -------------------------------
# Cluster and Sort Images
# -------------------------------
def cluster_and_rename_images(image_paths, features, output_dir, eps, min_samples, file_prefix, log_path):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(features)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} clusters.")

    cluster_map = defaultdict(list)
    for path, label in zip(image_paths, labels):
        cluster_map[label].append(path)

    # Sort within each cluster by similarity from oldest file
    ordered_paths = []
    log_entries = []
    for cluster_label in sorted(cluster_map.keys()):
        cluster_files = cluster_map[cluster_label]
        if not cluster_files:
            continue

        cluster_features = [features[image_paths.index(p)] for p in cluster_files]
        base_index = np.argmin([os.path.getctime(p) for p in cluster_files])
        ordered = [cluster_files[base_index]]
        remaining = list(range(len(cluster_files)))
        remaining.remove(base_index)

        while remaining:
            last = ordered[-1]
            last_feat = cluster_features[cluster_files.index(last)]
            similarities = [np.dot(last_feat, cluster_features[i]) / (
                np.linalg.norm(last_feat) * np.linalg.norm(cluster_features[i]) + 1e-5)
                for i in remaining]
            next_index = remaining[np.argmax(similarities)]
            ordered.append(cluster_files[next_index])
            remaining.remove(next_index)

        ordered_paths.extend([(path, cluster_label) for path in ordered])

    # Rename and copy files
    os.makedirs(output_dir, exist_ok=True)
    padding = len(str(len(ordered_paths)))

    for index, (original_path, cluster_label) in tqdm(enumerate(ordered_paths, start=1), total=len(ordered_paths), desc="Renaming clustered images"):
        extension = os.path.splitext(original_path)[1].lower()
        new_name = f"{file_prefix}{str(index).zfill(padding)}{extension}"
        target_path = os.path.join(output_dir, new_name)
        shutil.copy(original_path, target_path)
        log_entries.append((original_path, new_name, cluster_label))

    # Write CSV log
    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["original_path", "new_filename", "cluster_label"])
        writer.writerows(log_entries)

# -------------------------------
# Copy and Rename Videos
# -------------------------------
def copy_and_rename_videos(video_paths, output_dir, file_prefix):
    total_videos = len(video_paths)
    padding = len(str(total_videos))
    os.makedirs(output_dir, exist_ok=True)

    for index, video_path in tqdm(enumerate(video_paths, start=1), total=total_videos, desc="Copying videos"):
        extension = os.path.splitext(video_path)[1].lower()
        new_name = f"vid_{file_prefix}{str(index).zfill(padding)}{extension}"
        target_path = os.path.join(output_dir, new_name)
        shutil.copy(video_path, target_path)

# -------------------------------
# Main Pipeline
# -------------------------------
def run_lookalike_pipeline(input_dir, output_dir, eps, min_samples, use_pca, file_prefix, cache_dir, use_cache):
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
        print("[WARN] No images found in input folder.")

    if not video_paths:
        print("[WARN] No video files found in input folder.")

    # Process images
    if image_paths:
        model, transform = load_dinov2_model()
        feature_cache_path, path_cache_path = get_cache_paths(input_dir, cache_dir)
        cache_existed = os.path.exists(feature_cache_path) and os.path.exists(path_cache_path)

        if use_cache and cache_existed:
            print("[INFO] Using cached features...")
            with open(path_cache_path, 'r') as f:
                cached_paths = [line.strip() for line in f.readlines()]
            if cached_paths != image_paths:
                print("[WARN] Cached paths don't match current input. Recomputing features.")
                cache_existed = False

        if use_cache and cache_existed:
            features = np.load(feature_cache_path)
        else:
            features = extract_visual_features(image_paths, model, transform)
            np.save(feature_cache_path, features)
            with open(path_cache_path, 'w') as f:
                f.writelines([f"{p}\n" for p in image_paths])

        if use_pca:
            features = apply_pca(features, components=100)

        cluster_and_rename_images(
            image_paths,
            features,
            output_dir,
            eps,
            min_samples,
            file_prefix,
            os.path.join(output_dir, "log.csv")
        )

    # Process videos
    if video_paths:
        copy_and_rename_videos(video_paths, output_dir, file_prefix)

    print(f"[DONE] Processed {len(image_paths)} images and {len(video_paths)} videos into: {output_dir}")

# -------------------------------
# CLI Entry Point
# -------------------------------
def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Lookalike: Group similar images and copy videos using DINOv2 features.")
    parser.add_argument("--input", required=True, help="Input folder with images and videos")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN eps (cosine distance threshold)")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples per cluster")
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA reduction before clustering")
    parser.add_argument("--prefix", type=str, default="cluster", help="Prefix for renamed image/video files")
    parser.add_argument("--cache", type=str, help="Path to torch hub cache directory")
    parser.add_argument("--use_cache", action="store_true", help="Use cached features if available")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_arguments()
    run_lookalike_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        use_pca=args.use_pca,
        file_prefix=args.prefix,
        cache_dir=args.cache,
        use_cache=args.use_cache
    )
